"""
Vectorized Spherical Network Experiment

This is a standalone experiment script using the vectorized spherical network class
for significantly faster simulation (~10-50x speedup).

Mathematical model is identical to spherical_network_experiment.py.
"""

import numpy as np
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import vectorized network class
from LIF_objects.SphericalNeuronalNetworkVectorized import SphericalNeuronalNetworkVectorized

# Import visualization utilities (these work with the vectorized network)
from LIF_utils.criticality_analysis_utils import plot_enhanced_criticality_analysis
from LIF_utils.activity_vis_utils import (
    plot_psth_and_raster, plot_ei_psth_and_raster,
    plot_network_activation_percentage, plot_oscillation_frequency_analysis,
    plot_ei_frequency_analysis, plot_ei_synchrony_analysis,
    plot_inhibitory_effects_analysis, plot_stimulation_figure,
    plot_membrane_potential_traces
)
from LIF_utils.network_vis_utils import visualize_distance_weights_3d, visualize_ie_distance_weights_3d


def _build_stim_info_html(stim_info):
    """Build HTML for stimulation regime info panel."""
    if not stim_info:
        return '<div id="stimInfo"><span class="label">Stim:</span> None</div>'

    mode = stim_info.get('mode', 'unknown')
    interval = stim_info.get('interval')
    strength = stim_info.get('strength')
    fraction = stim_info.get('fraction', 0)

    lines = [f'<span class="label">Stim:</span> {mode}']

    if interval:
        lines.append(f'<span class="label">Interval:</span> {interval}ms')
    if strength:
        lines.append(f'<span class="label">Strength:</span> {strength}')
    if fraction:
        lines.append(f'<span class="label">Fraction:</span> {fraction*100:.0f}%')

    if mode == 'Current Injection':
        duration = stim_info.get('duration', 0)
        lines.append(f'<span class="label">Duration:</span> {duration} steps')
    elif mode == 'Poisson':
        prob = stim_info.get('probability', 0)
        dur = stim_info.get('poisson_duration', 1)
        lines.append(f'<span class="label">Prob:</span> {prob}')
        lines.append(f'<span class="label">Window:</span> {dur} steps')

    return '<div id="stimInfo">' + '<br>'.join(lines) + '</div>'


def create_threejs_3d_animation(network, activity_record, dt=0.1,
                                save_path="spherical_activity.html",
                                max_frames=500,
                                decay_factor=0.8,
                                sphere_opacity=0.05,
                                stimulation_record=None,
                                stim_info=None,
                                sparkle=True):
    """
    Create interactive 3D Three.js animation showing neural activity on a sphere.
    Allows smooth orbit/pan/zoom while animation plays.

    Parameters:
    -----------
    network : SphericalNeuronalNetworkVectorized
        The network with 3D positions
    activity_record : list of lists
        Each element contains indices of neurons that spiked at that timestep
    dt : float
        Timestep in ms
    save_path : str
        Path to save HTML file
    max_frames : int
        Maximum frames to include in animation (downsamples if needed)
    decay_factor : float
        How quickly activity fades (0.8 = 80% retained per frame)
    sphere_opacity : float
        Opacity of the translucent sphere outline
    stimulation_record : dict or None
        Dictionary with 'times' and 'neurons' keys tracking stimulation events
    stim_info : dict or None
        Stimulation regime info: {mode, interval, strength, duration, probability, fraction}
    sparkle : bool
        If True, enable audio sonification with crystalline pings for E/I spikes
    """
    import json

    print(f"\nCreating 3D Three.js animation...")

    # Extract positions
    positions = network.neuron_3d_positions
    n_neurons = network.n_neurons
    sphere_radius = network.sphere_radius

    # Create coordinate arrays
    x_coords = [positions[i][0] for i in range(n_neurons)]
    y_coords = [positions[i][1] for i in range(n_neurons)]
    z_coords = [positions[i][2] for i in range(n_neurons)]

    # Create inhibitory mask (using vectorized array)
    is_inhibitory = network.is_inhibitory.tolist()

    # Sample frames if too many
    total_frames = len(activity_record)
    print(f"Total simulation frames: {total_frames}")

    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        sampled_activity = [activity_record[i] for i in indices]
        print(f"Downsampled to {max_frames} frames for animation")
    else:
        sampled_activity = activity_record
        indices = np.arange(total_frames)

    # Build stimulation map: for each timestep, which neurons were stimulated
    stim_map = {}  # timestep -> set of neuron indices
    if stimulation_record and stimulation_record.get('times'):
        for stim_time, stim_neurons in zip(stimulation_record['times'], stimulation_record['neurons']):
            timestep = int(stim_time / dt)
            if timestep not in stim_map:
                stim_map[timestep] = set()
            stim_map[timestep].update(stim_neurons)

    # Pre-compute activity intensity for each frame with decay
    print("Pre-computing activity intensities...")
    activity_data = []
    stim_data = []
    current_intensity = np.zeros(n_neurons)
    current_stim_intensity = np.zeros(n_neurons)

    for frame_idx, active_neurons in enumerate(sampled_activity):
        current_intensity *= decay_factor
        current_stim_intensity *= decay_factor

        original_timestep = indices[frame_idx]
        stimulated_this_frame = stim_map.get(original_timestep, set())

        if len(active_neurons) > 0:
            current_intensity[active_neurons] = 1.0
            for neuron_idx in active_neurons:
                if neuron_idx in stimulated_this_frame:
                    current_stim_intensity[neuron_idx] = 1.0

        activity_data.append(current_intensity.copy().tolist())
        stim_data.append(current_stim_intensity.copy().tolist())

    # Build the HTML with embedded Three.js
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Spherical Neural Network Activity</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0a0a0f; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            z-index: 100;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 15px;
            background: rgba(0,0,0,0.5);
            padding: 10px 20px;
            border-radius: 5px;
        }}
        #controls button {{
            padding: 8px 16px;
            font-size: 14px;
            cursor: pointer;
            background: #333;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
        }}
        #controls button:hover {{ background: #444; }}
        #slider {{
            width: 400px;
            cursor: pointer;
        }}
        #timeDisplay {{
            color: white;
            font-family: monospace;
            min-width: 80px;
        }}
        #legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 12px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        #stimInfo {{
            position: absolute;
            top: 120px;
            right: 10px;
            color: #aaa;
            font-family: monospace;
            font-size: 11px;
            background: rgba(0,0,0,0.5);
            padding: 8px 10px;
            border-radius: 5px;
            line-height: 1.4;
        }}
        #stimInfo .label {{ color: #888; }}
    </style>
</head>
<body>
    <div id="info">3D Spherical Neural Network Activity<br>Drag to orbit, scroll to zoom</div>
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background: #ff3232;"></div> Excitatory</div>
        <div class="legend-item"><div class="legend-color" style="background: #3232ff;"></div> Inhibitory</div>
        <div class="legend-item"><div class="legend-color" style="background: #32ff32;"></div> Stimulated</div>
    </div>
    {_build_stim_info_html(stim_info)}
    <div id="controls">
        <button id="playPauseBtn">Pause</button>
        <button id="rotateBtn">Stop Rotate</button>
        {'<button id="soundBtn">Sound Off</button>' if sparkle else ''}
        <input type="range" id="slider" min="0" max="{len(sampled_activity)-1}" value="0">
        <span id="timeDisplay">0.0 ms</span>
        <button id="hideUiBtn">Hide UI</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data from Python
        const neuronPositions = {json.dumps(list(zip(x_coords, y_coords, z_coords)))};
        const isInhibitory = {json.dumps(is_inhibitory)};
        const activityData = {json.dumps(activity_data)};
        const stimData = {json.dumps(stim_data)};
        const sphereRadius = {sphere_radius};
        const sphereOpacity = {sphere_opacity};
        const dt = {dt};
        const frameIndices = {json.dumps(indices.tolist())};

        // Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0f);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(sphereRadius * 1.44, sphereRadius * 1.44, sphereRadius * 1.2);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        // Orbit controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = sphereRadius * 1.5;
        controls.maxDistance = sphereRadius * 10;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.69;

        // Create translucent sphere
        const sphereGeometry = new THREE.SphereGeometry(sphereRadius, 32, 32);
        const sphereMaterial = new THREE.MeshBasicMaterial({{
            color: 0x888888,
            transparent: true,
            opacity: sphereOpacity,
            wireframe: true
        }});
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        scene.add(sphere);

        // Create background neuron points (always visible, low opacity)
        const bgGeometry = new THREE.BufferGeometry();
        const neuronCount = neuronPositions.length;
        const bgPositions = new Float32Array(neuronCount * 3);
        const bgColors = new Float32Array(neuronCount * 3);

        for (let i = 0; i < neuronCount; i++) {{
            bgPositions[i * 3] = neuronPositions[i][0];
            bgPositions[i * 3 + 1] = neuronPositions[i][1];
            bgPositions[i * 3 + 2] = neuronPositions[i][2];

            if (isInhibitory[i]) {{
                bgColors[i * 3] = 0.15;
                bgColors[i * 3 + 1] = 0.15;
                bgColors[i * 3 + 2] = 0.4;
            }} else {{
                bgColors[i * 3] = 0.4;
                bgColors[i * 3 + 1] = 0.15;
                bgColors[i * 3 + 2] = 0.15;
            }}
        }}

        bgGeometry.setAttribute('position', new THREE.BufferAttribute(bgPositions, 3));
        bgGeometry.setAttribute('color', new THREE.BufferAttribute(bgColors, 3));

        const bgMaterial = new THREE.PointsMaterial({{
            size: 0.15,
            vertexColors: true,
            transparent: true,
            opacity: 0.3,
            sizeAttenuation: true
        }});

        const bgPoints = new THREE.Points(bgGeometry, bgMaterial);
        scene.add(bgPoints);

        // Create active neuron points (flashing when firing)
        const neuronGeometry = new THREE.BufferGeometry();

        const positions = new Float32Array(neuronCount * 3);
        const colors = new Float32Array(neuronCount * 3);
        const sizes = new Float32Array(neuronCount);
        const baseColors = [];

        for (let i = 0; i < neuronCount; i++) {{
            positions[i * 3] = neuronPositions[i][0];
            positions[i * 3 + 1] = neuronPositions[i][1];
            positions[i * 3 + 2] = neuronPositions[i][2];

            if (isInhibitory[i]) {{
                baseColors.push({{ r: 0.2, g: 0.2, b: 1.0 }});
            }} else {{
                baseColors.push({{ r: 1.0, g: 0.2, b: 0.2 }});
            }}

            colors[i * 3] = 0;
            colors[i * 3 + 1] = 0;
            colors[i * 3 + 2] = 0;
            sizes[i] = 0;
        }}

        neuronGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        neuronGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        neuronGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        // Custom shader for variable size points
        const neuronMaterial = new THREE.ShaderMaterial({{
            uniforms: {{
                pointTexture: {{ value: createCircleTexture() }}
            }},
            vertexShader: `
                attribute float size;
                varying vec3 vColor;
                void main() {{
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                uniform sampler2D pointTexture;
                varying vec3 vColor;
                void main() {{
                    vec4 texColor = texture2D(pointTexture, gl_PointCoord);
                    if (texColor.a < 0.1) discard;
                    gl_FragColor = vec4(vColor, texColor.a);
                }}
            `,
            blending: THREE.AdditiveBlending,
            depthTest: true,
            depthWrite: false,
            transparent: true,
            vertexColors: true
        }});

        const neuronPoints = new THREE.Points(neuronGeometry, neuronMaterial);
        scene.add(neuronPoints);

        function createCircleTexture() {{
            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.3, 'rgba(255,255,255,0.8)');
            gradient.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 64, 64);
            const texture = new THREE.CanvasTexture(canvas);
            return texture;
        }}

        // Animation state
        let currentFrame = 0;
        let isPlaying = true;
        let isRotating = true;
        let lastFrameTime = 0;
        const frameDelay = 33;
        const rotationSpeed = 0.002;

        // UI elements
        const slider = document.getElementById('slider');
        const timeDisplay = document.getElementById('timeDisplay');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const rotateBtn = document.getElementById('rotateBtn');

        playPauseBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';
        }});

        rotateBtn.addEventListener('click', () => {{
            isRotating = !isRotating;
            rotateBtn.textContent = isRotating ? 'Stop Rotate' : 'Rotate';
            controls.autoRotate = isRotating;
            controls.autoRotateSpeed = 0.69;
        }});

        // Hide UI functionality
        const hideUiBtn = document.getElementById('hideUiBtn');
        const infoPanel = document.getElementById('info');
        const legendPanel = document.getElementById('legend');
        const stimInfoPanel = document.getElementById('stimInfo');
        const controlsPanel = document.getElementById('controls');
        let uiHidden = false;

        hideUiBtn.addEventListener('click', () => {{
            uiHidden = !uiHidden;
            if (uiHidden) {{
                // Hide all UI elements except the button itself
                infoPanel.style.display = 'none';
                legendPanel.style.display = 'none';
                if (stimInfoPanel) stimInfoPanel.style.display = 'none';
                // Hide other controls but keep the button visible
                playPauseBtn.style.display = 'none';
                rotateBtn.style.display = 'none';
                slider.style.display = 'none';
                timeDisplay.style.display = 'none';
                if (document.getElementById('soundBtn')) {{
                    document.getElementById('soundBtn').style.display = 'none';
                }}
                // Halve sphere opacity
                sphereMaterial.opacity = sphereOpacity / 2;
                hideUiBtn.textContent = 'Show UI';
            }} else {{
                // Show all UI elements
                infoPanel.style.display = 'block';
                legendPanel.style.display = 'block';
                if (stimInfoPanel) stimInfoPanel.style.display = 'block';
                playPauseBtn.style.display = 'inline-block';
                rotateBtn.style.display = 'inline-block';
                slider.style.display = 'inline-block';
                timeDisplay.style.display = 'inline';
                if (document.getElementById('soundBtn')) {{
                    document.getElementById('soundBtn').style.display = 'inline-block';
                }}
                // Restore sphere opacity
                sphereMaterial.opacity = sphereOpacity;
                hideUiBtn.textContent = 'Hide UI';
            }}
        }});

        // Audio state and functions (sparkle mode)
        const sparkleEnabled = {'true' if sparkle else 'false'};
        let audioContext = null;
        let soundEnabled = false;
        let prevActivity = null;

        function initAudio() {{
            if (!audioContext) {{
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }}
        }}

        function playPing(isInhib) {{
            if (!soundEnabled || !audioContext) return;

            const osc = audioContext.createOscillator();
            const gain = audioContext.createGain();

            osc.frequency.value = isInhib ? 587 : 880;
            osc.type = 'sine';

            const now = audioContext.currentTime;
            gain.gain.setValueAtTime(0.015, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

            osc.connect(gain);
            gain.connect(audioContext.destination);
            osc.start(now);
            osc.stop(now + 0.12);
        }}

        if (sparkleEnabled) {{
            const soundBtn = document.getElementById('soundBtn');
            soundBtn.addEventListener('click', () => {{
                initAudio();
                soundEnabled = !soundEnabled;
                soundBtn.textContent = soundEnabled ? 'Sound On' : 'Sound Off';
            }});
        }}

        slider.addEventListener('input', (e) => {{
            currentFrame = parseInt(e.target.value);
            updateNeurons(currentFrame);
        }});

        function updateNeurons(frameIdx) {{
            const activity = activityData[frameIdx];
            const stim = stimData[frameIdx];
            const colorsAttr = neuronGeometry.attributes.color;
            const sizesAttr = neuronGeometry.attributes.size;

            for (let i = 0; i < neuronCount; i++) {{
                const intensity = activity[i];
                const stimIntensity = stim[i];

                if (sparkleEnabled && soundEnabled && intensity > 0.95) {{
                    const prevIntensity = prevActivity ? prevActivity[i] : 0;
                    if (prevIntensity < 0.9) {{
                        playPing(isInhibitory[i]);
                    }}
                }}

                if (intensity > 0.05) {{
                    if (stimIntensity > 0.05) {{
                        colorsAttr.array[i * 3] = 0.2 * stimIntensity;
                        colorsAttr.array[i * 3 + 1] = 1.0 * stimIntensity;
                        colorsAttr.array[i * 3 + 2] = 0.2 * stimIntensity;
                        sizesAttr.array[i] = 0.4 + stimIntensity * 0.8;
                    }} else {{
                        colorsAttr.array[i * 3] = baseColors[i].r * intensity;
                        colorsAttr.array[i * 3 + 1] = baseColors[i].g * intensity;
                        colorsAttr.array[i * 3 + 2] = baseColors[i].b * intensity;
                        sizesAttr.array[i] = 0.3 + intensity * 0.7;
                    }}
                }} else {{
                    colorsAttr.array[i * 3] = 0;
                    colorsAttr.array[i * 3 + 1] = 0;
                    colorsAttr.array[i * 3 + 2] = 0;
                    sizesAttr.array[i] = 0;
                }}
            }}

            prevActivity = activity;

            colorsAttr.needsUpdate = true;
            sizesAttr.needsUpdate = true;

            slider.value = frameIdx;
            const timeMs = frameIndices[frameIdx] * dt;
            timeDisplay.textContent = timeMs.toFixed(1) + ' ms';
        }}

        function animate(time) {{
            requestAnimationFrame(animate);

            if (isPlaying && time - lastFrameTime > frameDelay) {{
                currentFrame = (currentFrame + 1) % activityData.length;
                updateNeurons(currentFrame);
                lastFrameTime = time;
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        updateNeurons(0);
        animate(0);
    </script>
</body>
</html>'''

    with open(save_path, 'w') as f:
        f.write(html_content)

    print(f"3D animation saved to: {save_path}")


def run_vectorized_simulation(network, duration=1000.0, dt=0.1, stim_interval=None,
                              stim_interval_strength=10, stim_fraction=0.01,
                              stochastic_stim=False, no_stimulation=False,
                              current_injection_stimulation=True,
                              current_injection_duration=1,
                              poisson_process_stimulation=False,
                              poisson_process_probability=0.1,
                              poisson_process_duration=1,
                              track_neurons=None,
                              stim_location='uniform'):
    """
    Run simulation using vectorized network.

    This is a simplified version of run_unified_simulation optimized for the
    vectorized network class.
    """
    from tqdm import tqdm

    n_steps = int(duration / dt)
    activity_record = []
    stimulation_record = {'times': [], 'neurons': []}

    # Initialize neuron tracking data
    neuron_data = {}
    if track_neurons:
        for idx in track_neurons:
            neuron_data[idx] = {
                'v_history': [], 'g_e_history': [], 'g_i_history': [],
                'i_syn_history': [], 'spike_times': [], 'stim_times': [],
                'is_inhibitory': network.is_inhibitory[idx]
            }

    progress_bar = tqdm(total=n_steps, desc="Simulation", unit="steps")
    network.reset_all()

    # Track ongoing stimulations
    ongoing_current_injection = {}
    ongoing_poisson_window = {}

    # Pre-select fixed input neurons for interval-based stimulation (EXCITATORY ONLY)
    fixed_stim_neurons = None
    if not stochastic_stim and stim_interval and not no_stimulation:
        # Get indices of excitatory neurons only
        excitatory_indices = np.where(~network.is_inhibitory)[0]

        # Filter by location if top_down mode
        if stim_location == 'top_down':
            # Get y-coordinates for excitatory neurons (Y is up in Three.js visualization)
            y_coords = np.array([network.neuron_3d_positions[i][1] for i in excitatory_indices])
            # Find the top 10% threshold (90th percentile of y)
            y_threshold = np.percentile(y_coords, 90)
            # Filter to only neurons in top 10%
            top_mask = y_coords >= y_threshold
            excitatory_indices = excitatory_indices[top_mask]
            print(f"Top-down stimulation: selecting from {len(excitatory_indices)} excitatory neurons in top 10% of sphere (y >= {y_threshold:.2f})")

        n_stim = int(network.n_neurons * stim_fraction)
        # Make sure we don't try to select more neurons than available excitatory neurons
        n_stim = min(n_stim, len(excitatory_indices))
        fixed_stim_neurons = np.random.choice(excitatory_indices, size=n_stim, replace=False)
        print(f"Selected {len(fixed_stim_neurons)} EXCITATORY input neurons ({stim_fraction*100:.1f}% of network)")

    # Initial stochastic stim (excitatory only)
    if stochastic_stim and not no_stimulation:
        excitatory_indices = np.where(~network.is_inhibitory)[0]
        initial_stim_neurons = np.random.choice(excitatory_indices, size=min(5, len(excitatory_indices)), replace=False)
        stimulation_record['times'].append(0.0)
        stimulation_record['neurons'].append(list(initial_stim_neurons))
        for idx in initial_stim_neurons:
            network.stimulate_neuron(idx, current=15)

    # Main loop
    for step in range(n_steps):
        time_ms = step * dt

        if not no_stimulation:
            # Legacy stochastic stimulation (excitatory only)
            if stochastic_stim and np.random.random() < (dt / 150):
                excitatory_indices = np.where(~network.is_inhibitory)[0]
                current_strength = 20.0 + 10.0 * np.random.random()
                n_stim = min(int(network.n_neurons * stim_fraction), len(excitatory_indices))
                stim_neurons = np.random.choice(excitatory_indices, size=n_stim, replace=False)
                stimulation_record['times'].append(time_ms)
                stimulation_record['neurons'].append(list(stim_neurons))
                for idx in stim_neurons:
                    network.stimulate_neuron(idx, current=current_strength)

            # Interval-based stimulation
            if not stochastic_stim and stim_interval and step % int(stim_interval / dt) == 0 and step != 0:
                stim_neurons = fixed_stim_neurons

                if current_injection_stimulation:
                    for idx in stim_neurons:
                        ongoing_current_injection[idx] = current_injection_duration
                elif poisson_process_stimulation:
                    for idx in stim_neurons:
                        ongoing_poisson_window[idx] = poisson_process_duration

            # Process ongoing current injection
            if current_injection_stimulation and ongoing_current_injection:
                neurons_to_remove = []
                stimulated_this_step = []
                for idx in list(ongoing_current_injection.keys()):
                    steps_remaining = ongoing_current_injection[idx]
                    if steps_remaining > 0:
                        network.stimulate_neuron(idx, current=stim_interval_strength)
                        stimulated_this_step.append(idx)
                        ongoing_current_injection[idx] = steps_remaining - 1
                    if ongoing_current_injection[idx] <= 0:
                        neurons_to_remove.append(idx)
                if stimulated_this_step:
                    stimulation_record['times'].append(time_ms)
                    stimulation_record['neurons'].append(stimulated_this_step)
                for idx in neurons_to_remove:
                    del ongoing_current_injection[idx]

            # Process ongoing Poisson windows
            if poisson_process_stimulation and not current_injection_stimulation and ongoing_poisson_window:
                neurons_to_remove = []
                stimulated_this_step = []
                for idx in list(ongoing_poisson_window.keys()):
                    steps_remaining = ongoing_poisson_window[idx]
                    if steps_remaining > 0:
                        if np.random.random() < poisson_process_probability:
                            network.stimulate_neuron(idx, current=stim_interval_strength)
                            stimulated_this_step.append(idx)
                        ongoing_poisson_window[idx] = steps_remaining - 1
                    if ongoing_poisson_window[idx] <= 0:
                        neurons_to_remove.append(idx)
                if stimulated_this_step:
                    stimulation_record['times'].append(time_ms)
                    stimulation_record['neurons'].append(stimulated_this_step)
                for idx in neurons_to_remove:
                    del ongoing_poisson_window[idx]

        # Record state BEFORE update for tracked neurons
        if track_neurons:
            for idx in track_neurons:
                if idx in neuron_data:
                    neuron_data[idx]['v_history'].append(network.v[idx])
                    neuron_data[idx]['g_e_history'].append(network.g_e[idx])
                    neuron_data[idx]['g_i_history'].append(network.g_i[idx])
                    i_syn = (network.g_e[idx] * (network.e_reversal_arr[idx] - network.v[idx]) +
                             network.g_i[idx] * (network.i_reversal_arr[idx] - network.v[idx]))
                    neuron_data[idx]['i_syn_history'].append(i_syn)

        # Update network
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)

        # Record spikes for tracked neurons
        if track_neurons:
            for idx in track_neurons:
                if idx in neuron_data and idx in active_indices:
                    neuron_data[idx]['spike_times'].append(time_ms)

        progress_bar.update(1)

    progress_bar.close()
    return activity_record, neuron_data, stimulation_record


def save_experiment_config(network, experiment_params, save_path="experiment_config.json"):
    """Save experiment configuration to JSON."""
    # Get sample neuron parameters
    sample_exc_idx = np.where(~network.is_inhibitory)[0][0] if np.any(~network.is_inhibitory) else 0
    sample_inh_idx = np.where(network.is_inhibitory)[0][0] if np.any(network.is_inhibitory) else 0

    neuron_params = {
        "excitatory_neuron": {
            "v_rest": float(network.v_rest[sample_exc_idx]),
            "v_threshold": float(network.v_threshold[sample_exc_idx]),
            "v_reset": float(network.v_reset[sample_exc_idx]),
            "tau_m": float(network.tau_m[sample_exc_idx]),
            "tau_ref": float(network.tau_ref[sample_exc_idx]),
            "tau_e": float(network.tau_e[sample_exc_idx]),
            "tau_i": float(network.tau_i[sample_exc_idx]),
            "e_reversal": float(network.e_reversal_arr[sample_exc_idx]),
            "i_reversal": float(network.i_reversal_arr[sample_exc_idx]),
            "k_reversal": float(network.k_reversal[sample_exc_idx]),
            "v_noise_amp": float(network.v_noise_amp_arr[sample_exc_idx]),
            "i_noise_amp": float(network.i_noise_amp_arr[sample_exc_idx]),
            "adaptation_increment": float(network.adaptation_increment[sample_exc_idx]),
            "tau_adaptation": float(network.tau_adaptation[sample_exc_idx]),
            "is_inhibitory": False
        },
        "inhibitory_neuron": {
            "v_rest": float(network.v_rest[sample_inh_idx]),
            "v_threshold": float(network.v_threshold[sample_inh_idx]),
            "v_reset": float(network.v_reset[sample_inh_idx]),
            "tau_m": float(network.tau_m[sample_inh_idx]),
            "tau_ref": float(network.tau_ref[sample_inh_idx]),
            "tau_e": float(network.tau_e[sample_inh_idx]),
            "tau_i": float(network.tau_i[sample_inh_idx]),
            "e_reversal": float(network.e_reversal_arr[sample_inh_idx]),
            "i_reversal": float(network.i_reversal_arr[sample_inh_idx]),
            "k_reversal": float(network.k_reversal[sample_inh_idx]),
            "v_noise_amp": float(network.v_noise_amp_arr[sample_inh_idx]),
            "i_noise_amp": float(network.i_noise_amp_arr[sample_inh_idx]),
            "adaptation_increment": float(network.adaptation_increment[sample_inh_idx]),
            "tau_adaptation": float(network.tau_adaptation[sample_inh_idx]),
            "is_inhibitory": True
        }
    }

    n_excitatory = int(np.sum(~network.is_inhibitory))
    n_inhibitory = int(np.sum(network.is_inhibitory))

    network_params = {
        "n_neurons": network.n_neurons,
        "weight_scale": network.weight_scale,
        "distance_lambda": network.distance_lambda,
        "lambda_decay_ie": network.lambda_decay_ie,
        "sphere_radius": float(network.sphere_radius),
        "v_noise_amp": network.v_noise_amp,
        "i_noise_amp": network.i_noise_amp,
        "e_reversal": network.e_reversal,
        "i_reversal": network.i_reversal,
        "attempted_connections": network.attempted_connections,
        "nonzero_connections": network.nonzero_connections,
        "connection_density": network.nonzero_connections / (network.n_neurons * (network.n_neurons - 1))
    }

    config = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "3D Spherical network experiment (VECTORIZED)",
            "implementation": "SphericalNeuronalNetworkVectorized"
        },
        "experiment_parameters": experiment_params,
        "network_parameters": network_params,
        "neuron_parameters": neuron_params,
        "network_composition": {
            "total_neurons": network.n_neurons,
            "excitatory_neurons": n_excitatory,
            "inhibitory_neurons": n_inhibitory,
            "inhibitory_fraction_actual": n_inhibitory / network.n_neurons
        }
    }

    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nExperiment configuration saved to: {save_path}")
    return config


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_analysis_results(analysis_results, experiment_params=None, save_path="analysis_results.json"):
    """
    Save all analysis results with descriptive summaries to a JSON file.

    Each metric includes:
    - value: The computed metric value
    - summary: A plain-language explanation of what this metric means

    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing all analysis results from the simulation
    experiment_params : dict, optional
        Dictionary containing the experiment configuration parameters (predicted values)
    save_path : str
        Path to save the JSON file
    """

    # Build comprehensive results with summaries
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "Analysis results from vectorized spherical network simulation"
        },

        "simulation_performance": {
            "total_spikes": {
                "value": analysis_results.get("total_spikes", 0),
                "summary": "Total number of spikes (action potentials) recorded across all neurons during the entire simulation."
            },
            "avg_spikes_per_step": {
                "value": analysis_results.get("avg_spikes_per_step", 0),
                "summary": "Average number of neurons firing at each simulation timestep. Higher values indicate more synchronous or active networks."
            },
            "simulation_time_seconds": {
                "value": analysis_results.get("simulation_time", 0),
                "summary": "Wall-clock time in seconds to run the simulation."
            },
            "realtime_factor": {
                "value": analysis_results.get("realtime_factor", 0),
                "summary": "How many times faster than real-time the simulation ran. A value of 100x means 1 second of biological time was simulated in 0.01 seconds."
            }
        },

        "network_activation": {
            "mean_activation_percent": {
                "value": analysis_results.get("activation_data", {}).get("mean_pct", 0),
                "summary": "Average percentage of the network that was active (spiking) at any given moment. Low values (~1-5%) indicate sparse coding; high values (>20%) suggest excessive synchrony or runaway excitation."
            },
            "max_activation_percent": {
                "value": analysis_results.get("activation_data", {}).get("max_pct", 0),
                "summary": "Peak percentage of neurons firing simultaneously. Values approaching 100% indicate network-wide synchronization events."
            },
            "std_activation_percent": {
                "value": analysis_results.get("activation_data", {}).get("std_pct", 0),
                "summary": "Standard deviation of activation percentage. High variability suggests bursty/oscillatory dynamics; low variability indicates steady-state activity."
            }
        },

        "criticality_analysis": {
            "avalanche_count": {
                "value": analysis_results.get("criticality", {}).get("avalanche_count", 0),
                "summary": "Number of avalanches detected. An avalanche is a cascade of neural activity triggered by initial firing and propagating through the network until activity dies out."
            },
            "size_duration_scaling_exponent": {
                "value": analysis_results.get("criticality", {}).get("size_duration_scaling", 0),
                "summary": "The σ exponent relating avalanche size to duration (Size ~ Duration^σ). Critical systems show σ ≈ 1.5. Values < 1.5 suggest subcritical dynamics; values > 1.5 suggest supercritical dynamics."
            },
            "size_duration_r_squared": {
                "value": analysis_results.get("criticality", {}).get("size_duration_r_squared", 0),
                "summary": "R² goodness-of-fit for the size-duration scaling relationship. Values close to 1.0 indicate the power-law relationship is a good fit to the data."
            },
            "branching_ratio": {
                "value": analysis_results.get("criticality", {}).get("branching_ratio"),
                "summary": "Average number of neurons activated by each previously active neuron. Branching ratio = 1.0 indicates criticality (balanced propagation). <1.0 is subcritical (activity dies out); >1.0 is supercritical (activity explodes)."
            },
            "criticality_assessment": {
                "value": analysis_results.get("criticality", {}).get("assessment", "Unknown"),
                "summary": "Overall assessment of whether the network operates near the critical point, based on branching ratio and scaling exponents."
            }
        },

        "oscillation_frequency_analysis": {
            "peak_frequency_hz": {
                "value": analysis_results.get("oscillation_freq", {}).get("peak_frequency_hz", 0),
                "summary": "Dominant oscillation frequency in Hz from FFT analysis of total network activity. Common brain rhythms: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-100Hz)."
            },
            "dominant_band": {
                "value": analysis_results.get("oscillation_freq", {}).get("dominant_band", "Unknown"),
                "summary": "The frequency band containing the most power. Different bands are associated with different cognitive states and network dynamics."
            },
            "gamma_fraction": {
                "value": analysis_results.get("oscillation_freq", {}).get("gamma_fraction", 0),
                "summary": "Fraction of total spectral power in the gamma band (30-100Hz). High values indicate gamma-band oscillations, often associated with attention and binding in biological networks."
            },
            "is_gamma_oscillation": {
                "value": analysis_results.get("oscillation_freq", {}).get("is_gamma", False),
                "summary": "Whether the network exhibits gamma-band oscillations (30-100Hz). True if peak frequency is in gamma range or gamma power exceeds 30% of total."
            }
        },

        "ei_frequency_analysis": {
            "excitatory_peak_frequency_hz": {
                "value": analysis_results.get("ei_freq", {}).get("exc_peak_frequency_hz", 0),
                "summary": "Peak oscillation frequency of excitatory neuron population. Comparing E vs I peak frequencies reveals whether populations oscillate at the same or different frequencies."
            },
            "inhibitory_peak_frequency_hz": {
                "value": analysis_results.get("ei_freq", {}).get("inh_peak_frequency_hz", 0),
                "summary": "Peak oscillation frequency of inhibitory neuron population. Inhibitory interneurons often show faster intrinsic frequencies than excitatory neurons."
            },
            "exc_is_gamma": {
                "value": analysis_results.get("ei_freq", {}).get("exc_is_gamma", False),
                "summary": "Whether excitatory population oscillates in the gamma band."
            },
            "inh_is_gamma": {
                "value": analysis_results.get("ei_freq", {}).get("inh_is_gamma", False),
                "summary": "Whether inhibitory population oscillates in the gamma band."
            }
        },

        "ei_synchrony_analysis": {
            "excitatory_burst_index": {
                "value": analysis_results.get("sync_info", {}).get("exc_mean_burst_ref", 1.0),
                "summary": "Burst index for excitatory neurons (total spikes / unique neurons per bin). Value of 1.0 means each neuron fires once per bin (population synchrony). Values >1.0 indicate individual neurons fire multiple times per bin (bursting behavior)."
            },
            "inhibitory_burst_index": {
                "value": analysis_results.get("sync_info", {}).get("inh_mean_burst_ref", 1.0),
                "summary": "Burst index for inhibitory neurons. Higher values indicate more bursting behavior in the inhibitory population."
            },
            "reference_bin_ms": {
                "value": analysis_results.get("sync_info", {}).get("reference_bin_ms", 10),
                "summary": "Time bin size (ms) used for the reference burst index calculation. Larger bins can capture more refractory period violations."
            }
        },

        "inhibitory_effects_analysis": {
            "ie_connection_count": {
                "value": analysis_results.get("inh_info", {}).get("ie_connection_count", 0),
                "summary": "Number of inhibitory-to-excitatory (I→E) synaptic connections. These connections mediate feedforward and feedback inhibition of excitatory neurons."
            },
            "ii_connection_count": {
                "value": analysis_results.get("inh_info", {}).get("ii_connection_count", 0),
                "summary": "Number of inhibitory-to-inhibitory (I→I) synaptic connections. These connections mediate disinhibition and inhibitory network dynamics."
            },
            "ie_mean_weight": {
                "value": analysis_results.get("inh_info", {}).get("ie_mean_weight", 0),
                "summary": "Mean synaptic weight of I→E connections (negative values indicate inhibition). Stronger inhibition helps prevent runaway excitation."
            },
            "ii_mean_weight": {
                "value": analysis_results.get("inh_info", {}).get("ii_mean_weight", 0),
                "summary": "Mean synaptic weight of I→I connections. Self-inhibition among interneurons regulates the timing of inhibitory outputs."
            },
            "ie_per_target": {
                "value": analysis_results.get("inh_info", {}).get("ie_per_target", 0),
                "summary": "Average number of inhibitory inputs each excitatory neuron receives. Higher values mean stronger inhibitory control."
            },
            "ii_per_target": {
                "value": analysis_results.get("inh_info", {}).get("ii_per_target", 0),
                "summary": "Average number of inhibitory inputs each inhibitory neuron receives."
            },
            "e_neurons_tracked": {
                "value": analysis_results.get("inh_info", {}).get("e_neurons_tracked", 0),
                "summary": "Number of excitatory neurons whose conductance dynamics were tracked during simulation."
            },
            "e_neuron_mean_total_g_i": {
                "value": analysis_results.get("inh_info", {}).get("e_neuron_mean_total_g_i", 0),
                "summary": "Time-integrated mean inhibitory conductance received by tracked excitatory neurons. Indicates total inhibitory drive over the simulation."
            },
            "i_neurons_tracked": {
                "value": analysis_results.get("inh_info", {}).get("i_neurons_tracked", 0),
                "summary": "Number of inhibitory neurons whose conductance dynamics were tracked during simulation."
            },
            "i_neuron_mean_total_g_i": {
                "value": analysis_results.get("inh_info", {}).get("i_neuron_mean_total_g_i", 0),
                "summary": "Time-integrated mean inhibitory conductance received by tracked inhibitory neurons."
            }
        }
    }

    # Add experiment configuration (predicted/configured values) if provided
    if experiment_params is not None:
        results["experiment_configuration"] = {
            "description": "These are the configured/predicted parameters that were set before running the simulation.",
            "parameters": experiment_params
        }

    # Convert all numpy types to JSON-serializable Python types
    results = convert_to_json_serializable(results)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis results saved to: {save_path}")
    return results


def run_spherical_experiment_vectorized(
        n_neurons=100, connection_p=0.3, connection_probabilities=None,
        weight_scale=3.0,
        duration=5000.0, dt=0.1, stim_interval=None, stim_interval_strength=10,
        stim_fraction=0.01, transmission_delay=2.0, inhibitory_fraction=0.2,
        stochastic_stim=True, layout='sphere', no_stimulation=False,
        enable_noise=True, v_noise_amp=0.3, i_noise_amp=0.05,
        e_reversal=0.0, i_reversal=-80.0, random_seed=42, distance_lambda=0.1,
        lambda_decay_ie=0.05,
        animate=True, max_animation_frames=500,
        current_injection_stimulation=True,
        current_injection_duration=1,
        poisson_process_stimulation=False,
        poisson_process_probability=0.1,
        poisson_process_duration=1,
        jitter_v_rest=0.0, jitter_v_threshold=0.0,
        jitter_tau_m=0.0, jitter_tau_ref=0.0,
        jitter_tau_e=0.0, jitter_tau_i=0.0,
        jitter_adaptation_increment=0.0, jitter_tau_adaptation=0.0,
        stim_location='uniform'):
    """
    Run experiment with the vectorized 3D spherical network model.

    All parameters match run_spherical_experiment for compatibility.

    Parameters:
    -----------
    stim_location : str
        'uniform' - randomly sample stimulated neurons from all excitatory neurons
        'top_down' - sample only from excitatory neurons in the top 10% of the sphere (by y-coordinate, which is "up" in the 3D visualization)
    """
    print("Creating VECTORIZED 3D spherical neural network...")

    # Set random seed
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    # Set noise levels
    actual_v_noise = v_noise_amp if enable_noise else 0.0
    actual_i_noise = i_noise_amp if enable_noise else 0.0

    # Create vectorized network
    start_time = time.time()
    network = SphericalNeuronalNetworkVectorized(
        n_neurons=n_neurons,
        connection_p=connection_p,
        connection_probabilities=connection_probabilities,
        weight_scale=weight_scale,
        spatial=True,
        transmission_delay=transmission_delay,
        inhibitory_fraction=inhibitory_fraction,
        layout=layout,
        v_noise_amp=actual_v_noise,
        i_noise_amp=actual_i_noise,
        e_reversal=e_reversal,
        i_reversal=i_reversal,
        distance_lambda=distance_lambda,
        lambda_decay_ie=lambda_decay_ie,
        jitter_v_rest=jitter_v_rest,
        jitter_v_threshold=jitter_v_threshold,
        jitter_tau_m=jitter_tau_m,
        jitter_tau_ref=jitter_tau_ref,
        jitter_tau_e=jitter_tau_e,
        jitter_tau_i=jitter_tau_i,
        jitter_adaptation_increment=jitter_adaptation_increment,
        jitter_tau_adaptation=jitter_tau_adaptation
    )
    network_creation_time = time.time() - start_time
    print(f"Network creation time: {network_creation_time:.2f}s")

    # Generate network structure visualizations (before simulation)
    print("\nGenerating network structure visualizations...")
    print("Generating neuron parameter distributions...")
    network.plot_neuron_parameter_distributions(save_path="neuron_parameter_distributions_vectorized.png")
    print("Generating connection type distribution...")
    network.plot_connection_type_distribution(save_path="connection_type_distribution_vectorized.png")

    # Save experiment config
    experiment_params = {
        "n_neurons": n_neurons,
        "connection_p": connection_p,
        "connection_probabilities": connection_probabilities,
        "weight_scale": weight_scale,
        "duration": duration,
        "dt": dt,
        "stim_interval": stim_interval,
        "stim_interval_strength": stim_interval_strength,
        "stim_fraction": stim_fraction,
        "transmission_delay": transmission_delay,
        "inhibitory_fraction": inhibitory_fraction,
        "stochastic_stim": stochastic_stim,
        "layout": layout,
        "no_stimulation": no_stimulation,
        "enable_noise": enable_noise,
        "v_noise_amp": v_noise_amp,
        "i_noise_amp": i_noise_amp,
        "e_reversal": e_reversal,
        "i_reversal": i_reversal,
        "random_seed": random_seed,
        "distance_lambda": distance_lambda,
        "lambda_decay_ie": lambda_decay_ie,
        "animate": animate,
        "max_animation_frames": max_animation_frames,
        "current_injection_stimulation": current_injection_stimulation,
        "current_injection_duration": current_injection_duration,
        "poisson_process_stimulation": poisson_process_stimulation,
        "poisson_process_probability": poisson_process_probability,
        "poisson_process_duration": poisson_process_duration,
        "jitter_v_rest": jitter_v_rest,
        "jitter_v_threshold": jitter_v_threshold,
        "jitter_tau_m": jitter_tau_m,
        "jitter_tau_ref": jitter_tau_ref,
        "jitter_tau_e": jitter_tau_e,
        "jitter_tau_i": jitter_tau_i,
        "jitter_adaptation_increment": jitter_adaptation_increment,
        "jitter_tau_adaptation": jitter_tau_adaptation,
        "stim_location": stim_location,
        "implementation": "vectorized"
    }
    save_experiment_config(network, experiment_params, save_path="experiment_config_vectorized.json")

    print(f"\nStarting VECTORIZED simulation with {n_neurons} neurons for {duration} ms...")
    print(f"Sphere radius: {network.sphere_radius:.2f}")

    # Adjust stimulation
    actual_no_stimulation = no_stimulation
    actual_stim_interval = stim_interval if not no_stimulation else None
    actual_stochastic_stim = stochastic_stim if not no_stimulation else False

    # Build stim info
    stim_info = None
    if not actual_no_stimulation:
        if actual_stochastic_stim:
            stim_info = {'mode': 'Stochastic', 'fraction': stim_fraction}
        elif actual_stim_interval:
            if current_injection_stimulation:
                stim_info = {
                    'mode': 'Current Injection',
                    'interval': actual_stim_interval,
                    'strength': stim_interval_strength,
                    'duration': current_injection_duration,
                    'fraction': stim_fraction
                }
            elif poisson_process_stimulation:
                stim_info = {
                    'mode': 'Poisson',
                    'interval': actual_stim_interval,
                    'strength': stim_interval_strength,
                    'probability': poisson_process_probability,
                    'poisson_duration': poisson_process_duration,
                    'fraction': stim_fraction
                }

    # Select neurons to track (10 E + 10 I for inhibitory effects analysis)
    exc_indices = np.where(~network.is_inhibitory)[0]
    inh_indices = np.where(network.is_inhibitory)[0]
    track_exc = list(np.random.choice(exc_indices, size=min(10, len(exc_indices)), replace=False))
    track_inh = list(np.random.choice(inh_indices, size=min(10, len(inh_indices)), replace=False))
    track_neurons = track_exc + track_inh

    # Run simulation
    start_time = time.time()
    activity_record, neuron_data, stimulation_record = run_vectorized_simulation(
        network=network,
        duration=duration,
        dt=dt,
        stim_interval=actual_stim_interval,
        stim_interval_strength=stim_interval_strength,
        stim_fraction=stim_fraction,
        stochastic_stim=actual_stochastic_stim,
        no_stimulation=actual_no_stimulation,
        current_injection_stimulation=current_injection_stimulation,
        current_injection_duration=current_injection_duration,
        poisson_process_stimulation=poisson_process_stimulation,
        poisson_process_probability=poisson_process_probability,
        poisson_process_duration=poisson_process_duration,
        track_neurons=track_neurons,
        stim_location=stim_location
    )
    simulation_time = time.time() - start_time
    print(f"\nSimulation completed in {simulation_time:.2f}s")
    print(f"Simulation speed: {duration / simulation_time:.1f}x realtime")

    # Calculate activity statistics
    total_spikes = sum(len(active) for active in activity_record)
    avg_spikes_per_step = total_spikes / len(activity_record) if activity_record else 0
    print(f"Total spikes: {total_spikes}")
    print(f"Average spikes per timestep: {avg_spikes_per_step:.2f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Extract stim interval start times for plotting
    # stimulation_record['times'] contains every timestep where stim occurred,
    # but for visualization we want the interval START times (every stim_interval ms)
    # We compute these from the simulation parameters rather than the detailed record
    if stim_interval and not no_stimulation:
        # Stimulation starts at stim_interval, 2*stim_interval, etc.
        stim_interval_starts = list(np.arange(stim_interval, duration, stim_interval))
        # Calculate stimulation duration in ms for visualization
        if current_injection_stimulation:
            stim_duration_ms = current_injection_duration * dt
        elif poisson_process_stimulation:
            stim_duration_ms = poisson_process_duration * dt
        else:
            stim_duration_ms = None
        print(f"Passing {len(stim_interval_starts)} stimulation interval start times to plots (duration: {stim_duration_ms}ms)")
    else:
        stim_interval_starts = []
        stim_duration_ms = None
        print("No stimulation intervals to plot")

    # PSTH and raster plot
    print("Generating PSTH and raster plot...")
    plot_psth_and_raster(
        activity_record=activity_record,
        dt=dt,
        bin_size=10,
        stim_times=stim_interval_starts,
        stim_duration_ms=stim_duration_ms,
        save_path="psth_raster_plot_vectorized.png",
        darkstyle=True
    )

    # E/I PSTH and raster
    print("Generating E/I PSTH and raster plot...")
    plot_ei_psth_and_raster(
        network=network,
        activity_record=activity_record,
        dt=dt,
        bin_size=10,
        stim_times=stim_interval_starts,
        stim_duration_ms=stim_duration_ms,
        save_path="ei_psth_raster_plot_vectorized.png",
        darkstyle=True
    )

    # Network activation percentage
    print("Generating network activation plot...")
    plot_network_activation_percentage(
        activity_record=activity_record,
        n_neurons=n_neurons,
        dt=dt,
        stim_times=stim_interval_starts,
        stim_duration_ms=stim_duration_ms,
        save_path="network_activation_vectorized.png",
        darkstyle=True
    )

    # Criticality analysis (unstimulated periods only for accurate assessment)
    print("Generating criticality analysis (unstimulated periods only)...")
    criticality_info = plot_enhanced_criticality_analysis(
        network=network,
        save_path_prefix="avalanche_vectorized",
        darkstyle=True,
        activity_record=activity_record,
        dt=dt,
        stim_times=stim_interval_starts,
        stim_duration_ms=stim_duration_ms,
        buffer_ms=5.0
    )

    # Frequency analysis
    print("Generating oscillation frequency analysis...")
    _, freq_info = plot_oscillation_frequency_analysis(
        activity_record=activity_record,
        dt=dt,
        save_path="oscillation_frequency_analysis_vectorized.png",
        darkstyle=True
    )

    # E/I frequency analysis
    print("Generating E/I frequency analysis...")
    _, ei_freq_info = plot_ei_frequency_analysis(
        network=network,
        activity_record=activity_record,
        dt=dt,
        save_path="ei_frequency_analysis_vectorized.png",
        darkstyle=True
    )

    # E/I synchrony analysis
    print("Generating E/I synchrony analysis...")
    _, sync_info = plot_ei_synchrony_analysis(
        network=network,
        activity_record=activity_record,
        dt=dt,
        save_path="ei_synchrony_analysis_vectorized.png",
        darkstyle=True
    )

    print(f"\nE/I Synchrony Analysis Results:")
    print(f"  E mean burst index: {sync_info['exc_mean_burst_ref']:.2f}")
    print(f"  I mean burst index: {sync_info['inh_mean_burst_ref']:.2f}")

    # Inhibitory effects analysis
    print("\nGenerating inhibitory effects analysis...")
    _, inh_info = plot_inhibitory_effects_analysis(
        network=network,
        neuron_data=neuron_data,
        dt=dt,
        save_path="inhibitory_effects_analysis_vectorized.png",
        darkstyle=True
    )

    print(f"\nInhibitory Effects Analysis Results:")
    print(f"  I→E connections: {inh_info['ie_connection_count']}")
    print(f"  I→I connections: {inh_info['ii_connection_count']}")
    print(f"  I→E mean weight: {inh_info['ie_mean_weight']:.4f}")
    print(f"  I→I mean weight: {inh_info['ii_mean_weight']:.4f}")
    if inh_info.get('e_neurons_tracked', 0) > 0:
        print(f"  E neurons tracked: {inh_info['e_neurons_tracked']}")
        print(f"  Mean total inhibitory input to E: {inh_info.get('e_neuron_mean_total_g_i', 0):.2f}")
    if inh_info.get('i_neurons_tracked', 0) > 0:
        print(f"  I neurons tracked: {inh_info['i_neurons_tracked']}")
        print(f"  Mean total inhibitory input to I: {inh_info.get('i_neuron_mean_total_g_i', 0):.2f}")

    # Stimulation figure (only if stimulation was enabled)
    stim_fig_info = None
    if not actual_no_stimulation:
        print("\nGenerating comprehensive stimulation figure...")
        _, stim_fig_info = plot_stimulation_figure(
            network=network,
            neuron_data=neuron_data,
            stimulation_record=stimulation_record,
            activity_record=activity_record,
            dt=dt,
            stim_interval=actual_stim_interval,
            stim_strength=stim_interval_strength,
            save_path="stimulation_figure_vectorized.png",
            darkstyle=True
        )
        if stim_fig_info:
            print(f"\nStimulation Figure Summary:")
            print(f"  Total stim events: {stim_fig_info.get('total_stim_events', 0)}")
            print(f"  Unique neurons stimulated: {stim_fig_info.get('unique_neurons_stimulated', 0)}")
            print(f"  Stim neuron spikes: {stim_fig_info.get('stim_neuron_spikes', 0)}")
            print(f"  Cascade spikes: {stim_fig_info.get('cascade_spikes', 0)}")
            print(f"  Cascade fraction: {stim_fig_info.get('cascade_fraction', 0)*100:.1f}%")
            print(f"  Amplification: {stim_fig_info.get('mean_amplification', 0):.2f}x")

    # Membrane potential traces (high-res figure of selected neurons)
    print("\nGenerating membrane potential traces...")
    _, selected_neurons = plot_membrane_potential_traces(
        network=network,
        neuron_data=neuron_data,
        activity_record=activity_record,
        dt=dt,
        n_excitatory=3,
        n_inhibitory=3,
        save_path="membrane_potential_traces_vectorized.png",
        darkstyle=True,
        dpi=200,
        stim_times=stim_interval_starts,
        stim_duration_ms=stim_duration_ms
    )
    if selected_neurons:
        print(f"  Selected E neurons: {selected_neurons.get('excitatory', [])}")
        print(f"  Selected I neurons: {selected_neurons.get('inhibitory', [])}")


    # 3D distance-weight visualizations (E→E connections from an excitatory neuron)
    print("\nGenerating 3D E→E distance-weight visualization...")
    excitatory_indices = np.where(~network.is_inhibitory)[0]
    excitatory_neuron_idx = excitatory_indices[0] if len(excitatory_indices) > 0 else 0
    visualize_distance_weights_3d(
        network=network,
        neuron_idx=excitatory_neuron_idx,
        save_path="ee_distance_weights_3d_vectorized.html"
    )

    # I→E visualization
    print("Generating 3D I→E distance-weight visualization...")
    inhibitory_neuron_idx = np.where(network.is_inhibitory)[0][0] if np.any(network.is_inhibitory) else None
    if inhibitory_neuron_idx is not None:
        visualize_ie_distance_weights_3d(
            network=network,
            neuron_idx=inhibitory_neuron_idx,
            save_path="ie_distance_weights_3d_vectorized.html"
        )

    # 3D activity animation
    if animate:
        print("\nGenerating 3D activity animation...")
        create_threejs_3d_animation(
            network=network,
            activity_record=activity_record,
            dt=dt,
            save_path="spherical_activity_vectorized.html",
            max_frames=max_animation_frames,
            decay_factor=0.8,
            sphere_opacity=0.05,
            stimulation_record=stimulation_record,
            stim_info=stim_info,
            sparkle=True
        )

    # === Save comprehensive analysis results ===
    print("\nSaving comprehensive analysis results...")

    # Compute network activation statistics
    activation_pcts = [100 * len(active) / n_neurons for active in activity_record]
    activation_data = {
        'mean_pct': float(np.mean(activation_pcts)),
        'max_pct': float(np.max(activation_pcts)),
        'std_pct': float(np.std(activation_pcts))
    }

    # Extract criticality analysis (nested structure)
    criticality_data = {}
    if criticality_info and criticality_info.get('success'):
        crit_analysis = criticality_info.get('analysis', {})
        criticality_data = {
            'avalanche_count': crit_analysis.get('avalanche_count', 0),
            'branching_ratio': crit_analysis.get('branching_ratio'),
            'size_duration_scaling': crit_analysis.get('size_duration_exponent'),
            'size_duration_r_squared': crit_analysis.get('size_duration_r_squared'),
            'assessment': crit_analysis.get('assessment', 'Unknown')
        }

    # Collect all analysis results
    analysis_results = {
        # Simulation performance
        'total_spikes': total_spikes,
        'avg_spikes_per_step': avg_spikes_per_step,
        'simulation_time': simulation_time,
        'realtime_factor': duration / simulation_time if simulation_time > 0 else 0,

        # Network activation
        'activation_data': activation_data,

        # Criticality
        'criticality': criticality_data,

        # Oscillation frequency
        'oscillation_freq': freq_info if freq_info else {},

        # E/I frequency
        'ei_freq': ei_freq_info if ei_freq_info else {},

        # E/I synchrony
        'sync_info': sync_info if sync_info else {},

        # Inhibitory effects
        'inh_info': inh_info if inh_info else {},

        # Stimulation (if enabled)
        'stim_info': stim_fig_info if stim_fig_info else {}
    }

    save_analysis_results(analysis_results, experiment_params=experiment_params, save_path="analysis_results_vectorized.json")

    print("\n=== Vectorized Experiment Complete ===")
    print(f"Total time: {network_creation_time + simulation_time:.2f}s")

    return network, activity_record, neuron_data, stimulation_record


def run_biologically_plausible_simulation_vectorized(random_seed=42):
    """
    Run vectorized simulation with biologically plausible parameters.

    Parameters exactly match run_biologically_plausible_simulation() in
    spherical_network_experiment.py for comparison.
    """
    return run_spherical_experiment_vectorized(
        # Network parameters (same as circular experiment)
        n_neurons=6000, 
        connection_p=0.1,  # Fallback default
        connection_probabilities={  # Biological connection probabilities
            'ee': 0.12,   # E→E: ~10% (local recurrent excitation)
            'ei': 0.15,   # E→I: ~15-20% (feedforward to interneurons)
            'ie': 0.25,   # I→E: ~40-50% (strong blanket inhibition)
            'ii': 0.15,   # I→I: ~10-20% (interneuron networks)
        },
        weight_scale=0.60,
        inhibitory_fraction=0.2,  # Match original exactly
        transmission_delay=1,
        distance_lambda=0.20,  # Distance decay for all connection types (except I→E)
        lambda_decay_ie=0.15,  # Slower decay for I→E connections (longer range inhibition)

        # Simulation parameters
        duration=2500,  # ms
        dt=0.1,

        # Stimulation parameters
        stim_interval=500,  # the interval between stims
        stim_interval_strength=10,  # strength of each stim
        stim_fraction=0.10,  # fraction of total neurons to stimulate each interval
        no_stimulation=False,
        stochastic_stim=False,

        # curren inject stim mode parameters
        current_injection_stimulation=False,  # Sustained current for N timesteps
        current_injection_duration=200,  # How many timesteps current is applied
        
        # poisson stim mode parameters
        poisson_process_stimulation=True,  # Probabilistic stim (only if current_injection=False)
        poisson_process_probability=0.005,  # per timestep probability of stim in stim_fraction neurons
        poisson_process_duration=2000, #reduce back down to 150 or so

        # Noise parameters
        enable_noise=True,
        v_noise_amp=  0.1,
        i_noise_amp=  0.001,

        # Reversal potential parameters
        e_reversal=0.0,
        i_reversal=-80.0,

        # Other parameters
        layout='sphere',
        random_seed=random_seed,
        animate=True,
        max_animation_frames=2500,

        # Neuron parameter jitter
        # Voltage params: Gaussian (std dev in mV)
        jitter_v_rest=3.0,              # ±3mV Gaussian jitter on resting potential
        jitter_v_threshold=2.5,         # ±2.5mV Gaussian jitter on threshold
        # Time constants: Log-normal (coefficient of variation, CV = σ/μ)
        jitter_tau_m=0.3,               # 30% CV on membrane time constant
        jitter_tau_ref=0.1,             # 25% CV on refractory period
        jitter_tau_e=0.3,               # 30% CV on excitatory synaptic tau
        jitter_tau_i=0.3,               # 30% CV on inhibitory synaptic tau
        jitter_adaptation_increment=0.4,  # 40% CV on adaptation increment
        jitter_tau_adaptation=0.35,      # 35% CV on adaptation time constant

        # Stimulation location
        stim_location='top_down'  # 'uniform' or 'top_down' (top 10% of sphere)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("VECTORIZED Spherical Network Experiment")
    print("=" * 60)

    network, activity_record, neuron_data, stimulation_record = run_biologically_plausible_simulation_vectorized()
