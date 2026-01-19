import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import custom modules
from LIF_objects.SphericalNeuronalNetwork import SphericalNeuronalNetwork
from LIF_utils.simulation_utils import run_unified_simulation
from LIF_utils.criticality_analysis_utils import plot_enhanced_criticality_analysis
from LIF_utils.activity_vis_utils import plot_psth_and_raster, plot_ei_psth_and_raster, plot_network_activation_percentage, plot_oscillation_frequency_analysis, plot_ei_frequency_analysis, plot_ei_synchrony_analysis
from LIF_utils.network_vis_utils import visualize_distance_weights_3d, visualize_ie_distance_weights_3d


def save_experiment_config(network, experiment_params, save_path="experiment_config.json"):
    """
    Save a comprehensive configuration file detailing all network and neuron parameters.

    Parameters:
    -----------
    network : SphericalNeuronalNetwork
        The network object containing all configuration
    experiment_params : dict
        Dictionary of experiment-level parameters passed to run_spherical_experiment
    save_path : str
        Path to save the config file
    """
    # Get a sample neuron to extract class defaults
    sample_excitatory = None
    sample_inhibitory = None
    for neuron in network.neurons:
        if neuron.is_inhibitory and sample_inhibitory is None:
            sample_inhibitory = neuron
        elif not neuron.is_inhibitory and sample_excitatory is None:
            sample_excitatory = neuron
        if sample_excitatory and sample_inhibitory:
            break

    # Build neuron parameters from sample neurons
    neuron_params = {
        "excitatory_neuron": {
            "v_rest": sample_excitatory.v_rest,
            "v_threshold": sample_excitatory.v_threshold,
            "v_reset": sample_excitatory.v_reset,
            "tau_m": sample_excitatory.tau_m,
            "tau_ref": sample_excitatory.tau_ref,
            "tau_e": sample_excitatory.tau_e,
            "tau_i": sample_excitatory.tau_i,
            "e_reversal": sample_excitatory.e_reversal,
            "i_reversal": sample_excitatory.i_reversal,
            "k_reversal": sample_excitatory.k_reversal,
            "v_noise_amp": sample_excitatory.v_noise_amp,
            "i_noise_amp": sample_excitatory.i_noise_amp,
            "adaptation_increment": sample_excitatory.adaptation_increment,
            "tau_adaptation": sample_excitatory.tau_adaptation,
            "is_inhibitory": False
        } if sample_excitatory else None,
        "inhibitory_neuron": {
            "v_rest": sample_inhibitory.v_rest,
            "v_threshold": sample_inhibitory.v_threshold,
            "v_reset": sample_inhibitory.v_reset,
            "tau_m": sample_inhibitory.tau_m,
            "tau_ref": sample_inhibitory.tau_ref,
            "tau_e": sample_inhibitory.tau_e,
            "tau_i": sample_inhibitory.tau_i,
            "e_reversal": sample_inhibitory.e_reversal,
            "i_reversal": sample_inhibitory.i_reversal,
            "k_reversal": sample_inhibitory.k_reversal,
            "v_noise_amp": sample_inhibitory.v_noise_amp,
            "i_noise_amp": sample_inhibitory.i_noise_amp,
            "adaptation_increment": sample_inhibitory.adaptation_increment,
            "tau_adaptation": sample_inhibitory.tau_adaptation,
            "is_inhibitory": True
        } if sample_inhibitory else None
    }

    # Build network parameters
    network_params = {
        "n_neurons": network.n_neurons,
        "weight_scale": network.weight_scale,
        "distance_lambda": network.distance_lambda,
        "lambda_decay_ie": network.lambda_decay_ie,
        "sphere_radius": network.sphere_radius,
        "v_noise_amp": network.v_noise_amp,
        "i_noise_amp": network.i_noise_amp,
        "e_reversal": network.e_reversal,
        "i_reversal": network.i_reversal,
        "attempted_connections": network.attempted_connections,
        "nonzero_connections": network.nonzero_connections,
        "connection_density": network.nonzero_connections / (network.n_neurons * (network.n_neurons - 1))
    }

    # Count neuron types
    n_excitatory = sum(1 for n in network.neurons if not n.is_inhibitory)
    n_inhibitory = sum(1 for n in network.neurons if n.is_inhibitory)

    # Build complete config
    config = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "3D Spherical network experiment configuration"
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

    # Save to file
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nExperiment configuration saved to: {save_path}")
    return config


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
                                 sphere_opacity=0.2,
                                 stimulation_record=None,
                                 stim_info=None,
                                 sparkle=False):
    """
    Create interactive 3D Three.js animation showing neural activity on a sphere.
    Allows smooth orbit/pan/zoom while animation plays.

    Parameters:
    -----------
    network : SphericalNeuronalNetwork
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
    print(f"\nCreating 3D Three.js animation...")

    # Extract positions
    positions = network.neuron_3d_positions
    n_neurons = network.n_neurons
    sphere_radius = network.sphere_radius

    # Create coordinate arrays
    x_coords = [positions[i][0] for i in range(n_neurons)]
    y_coords = [positions[i][1] for i in range(n_neurons)]
    z_coords = [positions[i][2] for i in range(n_neurons)]

    # Create inhibitory mask
    is_inhibitory = [network.neurons[i].is_inhibitory for i in range(n_neurons)]

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
    # Also track stimulated neurons per frame
    print("Pre-computing activity intensities...")
    activity_data = []
    stim_data = []  # Track which neurons were stimulated at each frame
    current_intensity = np.zeros(n_neurons)
    current_stim_intensity = np.zeros(n_neurons)

    for frame_idx, active_neurons in enumerate(sampled_activity):
        current_intensity *= decay_factor
        current_stim_intensity *= decay_factor

        # Get original timestep for this frame
        original_timestep = indices[frame_idx]

        # Check if any neurons were stimulated at this timestep
        stimulated_this_frame = stim_map.get(original_timestep, set())

        if len(active_neurons) > 0:
            current_intensity[active_neurons] = 1.0
            # Mark stimulated neurons that are also firing
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

            // Dim colors: red for excitatory, blue for inhibitory
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

            // Base colors: red for excitatory, blue for inhibitory
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
        const frameDelay = 33; // ~30fps
        const rotationSpeed = 0.002; // Very slow rotation speed

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
            controls.autoRotateSpeed = 0.69; // Very slow rotation
        }});

        // Audio state and functions (sparkle mode)
        const sparkleEnabled = {'true' if sparkle else 'false'};
        let audioContext = null;
        let soundEnabled = false;
        let prevActivity = null;  // Track previous frame to detect new spikes

        function initAudio() {{
            if (!audioContext) {{
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }}
        }}

        function playPing(isInhib) {{
            if (!soundEnabled || !audioContext) return;

            const osc = audioContext.createOscillator();
            const gain = audioContext.createGain();

            // Excitatory: 880Hz (A5), Inhibitory: 587Hz (D5, perfect 5th below)
            osc.frequency.value = isInhib ? 587 : 880;
            osc.type = 'sine';  // Pure tone for crystalline sound

            // Quick attack, short decay for crystalline ping
            const now = audioContext.currentTime;
            gain.gain.setValueAtTime(0.015, now);  // Lower volume to prevent clipping with many pings
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

                // Sparkle: detect new spikes (intensity jumped to ~1.0 from lower value)
                if (sparkleEnabled && soundEnabled && intensity > 0.95) {{
                    const prevIntensity = prevActivity ? prevActivity[i] : 0;
                    if (prevIntensity < 0.9) {{
                        // This is a new spike - play ping
                        playPing(isInhibitory[i]);
                    }}
                }}

                if (intensity > 0.05) {{
                    // Check if this neuron was stimulated - show as green
                    if (stimIntensity > 0.05) {{
                        // Green for stimulated neurons
                        colorsAttr.array[i * 3] = 0.2 * stimIntensity;
                        colorsAttr.array[i * 3 + 1] = 1.0 * stimIntensity;
                        colorsAttr.array[i * 3 + 2] = 0.2 * stimIntensity;
                        sizesAttr.array[i] = 0.4 + stimIntensity * 0.8;  // Slightly larger
                    }} else {{
                        // Normal color based on neuron type
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

            // Store current activity for next frame comparison
            prevActivity = activity;

            colorsAttr.needsUpdate = true;
            sizesAttr.needsUpdate = true;

            // Update UI
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

        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Start
        updateNeurons(0);
        animate(0);
    </script>
</body>
</html>'''

    with open(save_path, 'w') as f:
        f.write(html_content)

    print(f"3D animation saved to: {save_path}")


def create_plotly_3d_animation(network, activity_record, dt=0.1,
                                save_path="spherical_activity.html",
                                max_frames=500,
                                decay_factor=0.8,
                                sphere_opacity=0.1):
    """
    Create interactive 3D Plotly animation showing neural activity on a sphere.

    Parameters:
    -----------
    network : SphericalNeuronalNetwork
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

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The figure object
    """
    print(f"\nCreating 3D Plotly animation...")

    # Extract positions
    positions = network.neuron_3d_positions
    n_neurons = network.n_neurons
    sphere_radius = network.sphere_radius

    # Create coordinate arrays
    x_coords = np.array([positions[i][0] for i in range(n_neurons)])
    y_coords = np.array([positions[i][1] for i in range(n_neurons)])
    z_coords = np.array([positions[i][2] for i in range(n_neurons)])

    # Create inhibitory mask for coloring
    is_inhibitory = np.array([network.neurons[i].is_inhibitory for i in range(n_neurons)])

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

    # Pre-compute activity intensity for each frame with decay
    print("Pre-computing activity intensities...")
    activity_intensity = np.zeros((len(sampled_activity), n_neurons))
    current_intensity = np.zeros(n_neurons)

    for frame_idx, active_neurons in enumerate(sampled_activity):
        current_intensity *= decay_factor  # Decay previous activity
        if len(active_neurons) > 0:
            current_intensity[active_neurons] = 1.0  # New spikes
        activity_intensity[frame_idx] = current_intensity.copy()

    # Create translucent sphere surface
    print("Creating sphere surface mesh...")
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    sphere_x = sphere_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = sphere_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create figure
    fig = go.Figure()

    # Add translucent sphere outline
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=sphere_opacity,
        colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
        showscale=False,
        name='Sphere Boundary',
        hoverinfo='skip'
    ))

    # Create frames for animation
    print("Building animation frames...")
    frames = []

    for frame_idx in range(len(sampled_activity)):
        intensity = activity_intensity[frame_idx]

        # Only show neurons with intensity > threshold
        visible_mask = intensity > 0.05

        if np.sum(visible_mask) > 0:
            visible_indices = np.where(visible_mask)[0]

            # Set colors with opacity encoded in RGBA format
            opacities = intensity[visible_mask]
            colors = [f'rgba(255,50,50,{op:.2f})' if not is_inhibitory[i] else f'rgba(50,50,255,{op:.2f})'
                      for i, op in zip(visible_indices, opacities)]

            # Scale marker size with intensity
            sizes = 4 + intensity[visible_mask] * 8

            frame_data = [
                go.Surface(
                    x=sphere_x, y=sphere_y, z=sphere_z,
                    opacity=sphere_opacity,
                    colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
                    showscale=False,
                    hoverinfo='skip'
                ),
                go.Scatter3d(
                    x=x_coords[visible_mask],
                    y=y_coords[visible_mask],
                    z=z_coords[visible_mask],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        line=dict(width=0)
                    ),
                    name='Active Neurons',
                    hovertemplate='Neuron %{text}<br>Intensity: %{customdata:.2f}<extra></extra>',
                    text=[str(i) for i in visible_indices],
                    customdata=opacities
                )
            ]
        else:
            # No active neurons - just show sphere
            frame_data = [
                go.Surface(
                    x=sphere_x, y=sphere_y, z=sphere_z,
                    opacity=sphere_opacity,
                    colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
                    showscale=False,
                    hoverinfo='skip'
                ),
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='markers',
                    marker=dict(size=1),
                    name='Active Neurons'
                )
            ]

        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=dict(text=f"3D Spherical Neural Network - Time: {indices[frame_idx] * dt:.1f} ms")
            )
        ))

    # Add initial frame data (frame 0)
    intensity = activity_intensity[0]
    visible_mask = intensity > 0.05

    if np.sum(visible_mask) > 0:
        visible_indices = np.where(visible_mask)[0]
        opacities = intensity[visible_mask]
        colors = [f'rgba(255,50,50,{op:.2f})' if not is_inhibitory[i] else f'rgba(50,50,255,{op:.2f})'
                  for i, op in zip(visible_indices, opacities)]
        sizes = 4 + intensity[visible_mask] * 8

        fig.add_trace(go.Scatter3d(
            x=x_coords[visible_mask],
            y=y_coords[visible_mask],
            z=z_coords[visible_mask],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors
            ),
            name='Active Neurons'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=1),
            name='Active Neurons'
        ))

    fig.frames = frames

    # Create slider steps
    slider_steps = []
    step_interval = max(1, len(frames) // 50)  # Show ~50 ticks on slider
    for i in range(0, len(frames), step_interval):
        slider_steps.append(dict(
            args=[[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            method="animate",
            label=f"{indices[i] * dt:.0f}"
        ))

    # Update layout with animation controls
    fig.update_layout(
        title=dict(
            text="3D Spherical Neural Network Activity",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            aspectmode='cube',
            xaxis=dict(range=[-sphere_radius * 1.2, sphere_radius * 1.2], showgrid=True, gridcolor='rgba(80,80,80,0.015)', showticklabels=False, showline=True, linecolor='rgba(80,80,80,0.015)', zeroline=False, showbackground=False),
            yaxis=dict(range=[-sphere_radius * 1.2, sphere_radius * 1.2], showgrid=True, gridcolor='rgba(80,80,80,0.015)', showticklabels=False, showline=True, linecolor='rgba(80,80,80,0.015)', zeroline=False, showbackground=False),
            zaxis=dict(range=[-sphere_radius * 1.2, sphere_radius * 1.2], showgrid=True, gridcolor='rgba(80,80,80,0.015)', showticklabels=False, showline=True, linecolor='rgba(80,80,80,0.015)', zeroline=False, showbackground=False),
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.9)),
            bgcolor='rgba(10,10,15,1)',
            uirevision='constant'
        ),
        uirevision='constant',
        paper_bgcolor='rgba(30,30,40,1)',
        font=dict(color='white'),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.1,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 0, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                prefix="Time: ",
                suffix=" ms",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=slider_steps
        )],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(50,50,60,0.8)'
        ),
        annotations=[
            dict(
                text="<span style='color:red'>Red = Excitatory</span> | <span style='color:blue'>Blue = Inhibitory</span>",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                font=dict(size=12)
            )
        ]
    )

    # Save to HTML
    print(f"Saving animation to {save_path}...")
    fig.write_html(save_path, include_plotlyjs=True, full_html=True)

    # Inject JavaScript to auto pause/play for correct framerate
    with open(save_path, 'r') as f:
        html_content = f.read()

    auto_play_script = """
<script>
window.addEventListener('load', function() {
    setTimeout(function() {
        var plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv) {
            // First pause
            Plotly.animate(plotDiv, [], {
                frame: {duration: 0, redraw: false},
                mode: 'immediate'
            });
            // Then play with correct settings after a brief delay
            setTimeout(function() {
                Plotly.animate(plotDiv, null, {
                    frame: {duration: 0, redraw: true},
                    transition: {duration: 0},
                    fromcurrent: true,
                    mode: 'immediate'
                });
            }, 100);
        }
    }, 500);
});
</script>
</body>
"""
    html_content = html_content.replace('</body>', auto_play_script)

    with open(save_path, 'w') as f:
        f.write(html_content)

    print(f"3D animation saved to: {save_path}")

    return fig


def run_spherical_experiment(n_neurons=100, connection_p=0.3, connection_probabilities=None,
                              weight_scale=3.0,
                              duration=5000.0, dt=0.1, stim_interval=None, stim_interval_strength=10,
                              stim_fraction=0.01, transmission_delay=2.0, inhibitory_fraction=0.2,
                              stochastic_stim=True, layout='sphere', no_stimulation=False,
                              enable_noise=True, v_noise_amp=0.3, i_noise_amp=0.05,
                              e_reversal=0.0, i_reversal=-80.0, random_seed=42, distance_lambda=0.1,
                              lambda_decay_ie=0.05,
                              animate=True, max_animation_frames=500,
                              # Stimulation mode parameters (only active when stochastic_stim=False and no_stimulation=False)
                              current_injection_stimulation=True,  # Sustained current for N timesteps
                              current_injection_duration=1,        # How many timesteps current is applied
                              poisson_process_stimulation=False,   # Probabilistic stim (only if current_injection=False)
                              poisson_process_probability=0.1,     # Per-timestep probability of firing during window
                              poisson_process_duration=1,          # Window length in timesteps
                              # Neuron parameter jitter (0 = no jitter)
                              # Voltage params: Gaussian std dev (mV)
                              jitter_v_rest=0.0,                   # Resting potential jitter (mV, Gaussian)
                              jitter_v_threshold=0.0,              # Threshold jitter (mV, Gaussian)
                              # Time constants: Coefficient of variation (CV = σ/μ, log-normal)
                              jitter_tau_m=0.0,                    # Membrane time constant CV (log-normal)
                              jitter_tau_ref=0.0,                  # Refractory period CV (log-normal)
                              jitter_tau_e=0.0,                    # Excitatory synaptic tau CV (log-normal)
                              jitter_tau_i=0.0,                    # Inhibitory synaptic tau CV (log-normal)
                              jitter_adaptation_increment=0.0,     # Adaptation increment CV (log-normal)
                              jitter_tau_adaptation=0.0):          # Adaptation time constant CV (log-normal)
    """
    Run experiment with the 3D spherical network model.

    Parameters:
    -----------
    n_neurons : int
        Number of neurons in the network
    connection_p : float
        Default connection probability between neurons (0-1). Used as fallback
        when connection_probabilities is not specified or partially specified.
    connection_probabilities : dict or None
        Per-connection-type probabilities. If provided, overrides connection_p.
        Keys: 'ee' (E→E), 'ei' (E→I), 'ie' (I→E), 'ii' (I→I)
        Example biological values: {'ee': 0.1, 'ei': 0.15, 'ie': 0.4, 'ii': 0.15}
    weight_scale : float
        Scale factor for synaptic weights
    duration : float
        Duration of simulation in ms
    dt : float
        Time step size in ms
    stim_interval : float or None
        Interval for regular stimulation in ms (None for stochastic)
    stim_interval_strength : float
        Strength of stimulation current
    stim_fraction : float
        Fraction of neurons to stimulate
    transmission_delay : float
        Synaptic transmission delay in ms
    inhibitory_fraction : float
        Fraction of neurons that are inhibitory (0-1)
    stochastic_stim : bool
        Whether to use stochastic stimulation
    layout : str
        Spatial layout: 'sphere' (volume) or 'sphere-surface'
    no_stimulation : bool
        If True, no external stimulation will be applied during the simulation
    enable_noise : bool
        If True, apply biological noise to membrane potential and synaptic input
    v_noise_amp : float
        Amplitude of membrane potential noise in mV (if enable_noise is True)
    i_noise_amp : float
        Amplitude of synaptic current noise (if enable_noise is True)
    e_reversal : float
        Excitatory reversal potential (mV)
    i_reversal : float
        Inhibitory reversal potential (mV)
    random_seed : int
        Seed for random number generation to ensure reproducibility
    distance_lambda : float
        Distance decay constant for synaptic weights (higher values mean faster decay)
    lambda_decay_ie : float
        Distance decay constant specifically for inhibitory→excitatory connections
        (default 0.05, meaning slower decay / longer range for I→E connections)
    animate : bool
        If True, generate a 3D Plotly animation of the neural activity
    max_animation_frames : int
        Maximum number of frames for the animation (downsamples if exceeded)
    current_injection_stimulation : bool
        Sustained current mode: apply current for N consecutive timesteps.
        Only used when stochastic_stim=False and no_stimulation=False.
    current_injection_duration : int
        Number of timesteps current is applied per stim_interval trigger
    poisson_process_stimulation : bool
        Probabilistic mode: each timestep has probability P of triggering stim.
        Only used when current_injection=False, stochastic_stim=False, no_stimulation=False.
    poisson_process_probability : float
        Per-timestep probability (0-1) of stimulation during the window.
        When triggered, applies stim_interval_strength current.
    poisson_process_duration : int
        Window length (timesteps) for Poisson process per stim_interval trigger
    """
    print("Creating 3D spherical neural network with reversal potentials...")

    # Set the initial random seed for network creation
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed} for reproducible results")

    # Set actual noise levels based on enable_noise flag
    actual_v_noise = v_noise_amp if enable_noise else 0.0
    actual_i_noise = i_noise_amp if enable_noise else 0.0

    # Create network with the 3D spherical model
    network = SphericalNeuronalNetwork(
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
        # Neuron parameter jitter
        jitter_v_rest=jitter_v_rest,
        jitter_v_threshold=jitter_v_threshold,
        jitter_tau_m=jitter_tau_m,
        jitter_tau_ref=jitter_tau_ref,
        jitter_tau_e=jitter_tau_e,
        jitter_tau_i=jitter_tau_i,
        jitter_adaptation_increment=jitter_adaptation_increment,
        jitter_tau_adaptation=jitter_tau_adaptation
    )

    # Generate connection type distribution figure
    network.plot_connection_type_distribution()

    # Generate neuron parameter distribution histograms (shows jitter effects)
    print("\nGenerating neuron parameter distribution histograms...")
    network.plot_neuron_parameter_distributions(
        save_path="neuron_parameter_distributions.png",
        darkstyle=True
    )

    # Save experiment configuration
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
        "jitter_tau_adaptation": jitter_tau_adaptation
    }
    save_experiment_config(network, experiment_params, save_path="experiment_config.json")

    print(f"\nStarting 3D spherical simulation with {n_neurons} neurons for {duration} ms...")
    print(f"Sphere radius: {network.sphere_radius:.2f}")
    print(f"Network parameters: connection_p={connection_p}, weight_scale={weight_scale}")
    print(f"Transmission delay: {transmission_delay} ms")
    print(f"Inhibitory fraction: {inhibitory_fraction * 100:.1f}%")
    print(f"Reversal potentials: E={e_reversal}mV, I={i_reversal}mV")

    # Print noise information
    if enable_noise:
        print(f"Neural noise enabled: membrane={v_noise_amp} mV, synaptic={i_noise_amp}")
    else:
        print("Neural noise disabled")

    # Adjust stimulation parameters based on no_stimulation flag
    if no_stimulation:
        print("Stimulation: NONE (no external stimulation)")
        stochastic_stim = False
        stim_interval = None
    else:
        print(f"Stimulation: {'Stochastic' if stochastic_stim else 'Regular'}")
        if stim_interval:
            print(f"Stimulation interval: {stim_interval}ms, strength: {stim_interval_strength}")

    # Print stimulation mode info
    if not no_stimulation:
        if current_injection_stimulation:
            print(f"Stimulation mode: Current Injection (duration={current_injection_duration} timesteps)")
        elif poisson_process_stimulation:
            print(f"Stimulation mode: Poisson Process (probability={poisson_process_probability}, duration={poisson_process_duration} timesteps)")

    # Run simulation
    print("\n=== Running simulation ===")
    activity_record, neuron_data, stimulation_record = run_unified_simulation(
        network,
        duration=duration,
        dt=dt,
        stim_interval=stim_interval,
        stim_interval_strength=stim_interval_strength,
        stim_fraction=stim_fraction,
        stim_neuron=None,
        track_neurons=None,
        stochastic_stim=stochastic_stim,
        no_stimulation=no_stimulation,
        current_injection_stimulation=current_injection_stimulation,
        current_injection_duration=current_injection_duration,
        poisson_process_stimulation=poisson_process_stimulation,
        poisson_process_probability=poisson_process_probability,
        poisson_process_duration=poisson_process_duration
    )

    # Analyze avalanches
    print("\nAnalyzing avalanche statistics:")
    avalanche_sizes = network.avalanche_sizes
    avalanche_durations = network.avalanche_durations
    print(f"Detected {len(avalanche_sizes)} avalanches")

    # Generate criticality analysis
    print("\nGenerating criticality analysis...")
    criticality_results = plot_enhanced_criticality_analysis(
        network,
        save_path_prefix="avalanche",
        darkstyle=True
    )

    if criticality_results["success"]:
        analysis = criticality_results["analysis"]
        print(f"\nAnalyzed {analysis['avalanche_count']} avalanches.")
        print(f"Assessment: {analysis['assessment']}")
        if analysis['is_critical']:
            print("The network exhibits critical behavior!")
    else:
        print(f"Criticality analysis: {criticality_results['message']}")

    # Extract stimulation times for plots
    stim_times = stimulation_record.get('times', []) if stimulation_record else []

    # Build stim_info for HTML display
    if no_stimulation:
        stim_info = {'mode': 'None'}
    elif current_injection_stimulation:
        stim_info = {
            'mode': 'Current Injection',
            'interval': stim_interval,
            'strength': stim_interval_strength,
            'fraction': stim_fraction,
            'duration': current_injection_duration
        }
    elif poisson_process_stimulation:
        stim_info = {
            'mode': 'Poisson',
            'interval': stim_interval,
            'strength': stim_interval_strength,
            'fraction': stim_fraction,
            'probability': poisson_process_probability,
            'poisson_duration': poisson_process_duration
        }
    else:
        stim_info = {'mode': 'Stochastic' if stochastic_stim else 'Unknown'}

    # Generate 3D Plotly animation
    if animate:
        print("\n=== Generating 3D visualization ===")
        create_threejs_3d_animation(
            network,
            activity_record,
            dt=dt,
            save_path="spherical_activity.html",
            max_frames=max_animation_frames,
            decay_factor=0.8,
            sphere_opacity=0.05,
            stimulation_record=stimulation_record,
            stim_info=stim_info,
            sparkle=True
        )

    # Generate PSTH and raster plot
    print("\nGenerating PSTH and raster plot...")
    plot_psth_and_raster(
        activity_record,
        stim_times=stim_times if stim_times else None,
        bin_size=20,  # 20 timesteps per bin = 2ms at dt=0.1
        dt=dt,
        neuron_subset=None,  # Use all neurons
        figsize=(14, 10),
        save_path="psth_raster_plot.png",
        darkstyle=True
    )

    # Generate E/I separated PSTH and raster plot
    print("\nGenerating excitatory/inhibitory PSTH and raster plot...")
    plot_ei_psth_and_raster(
        network=network,
        activity_record=activity_record,
        bin_size=20,  # 20 timesteps per bin = 2ms at dt=0.1
        dt=dt,
        figsize=(14, 12),
        save_path="ei_psth_raster_plot.png",
        darkstyle=True,
        stim_times=stim_times if stim_times else None
    )

    # Generate network activation percentage plot
    print("\nGenerating network activation percentage plot...")
    plot_network_activation_percentage(
        activity_record=activity_record,
        n_neurons=n_neurons,
        dt=dt,
        stim_times=stim_times if stim_times else None,
        window_ms=5.0,
        save_path="network_activation.png",
        figsize=(14, 6),
        darkstyle=True
    )

    # Generate oscillation frequency analysis (gamma detection)
    print("\nGenerating oscillation frequency analysis (gamma detection)...")
    freq_fig, freq_info = plot_oscillation_frequency_analysis(
        activity_record=activity_record,
        dt=dt,
        figsize=(14, 10),
        dpi=150,
        save_path="oscillation_frequency_analysis.png",
        darkstyle=True
    )

    # Print gamma detection results
    print(f"\nFrequency Analysis Results:")
    print(f"  Peak frequency: {freq_info['peak_frequency_hz']:.1f} Hz")
    print(f"  Dominant band: {freq_info['dominant_band']}")
    print(f"  Gamma fraction: {freq_info['gamma_fraction']:.2%}")
    print(f"  Gamma detected: {'YES' if freq_info['is_gamma'] else 'NO'}")

    # Generate E/I frequency analysis (separate spectra for excitatory and inhibitory)
    print("\nGenerating E/I frequency analysis...")
    ei_freq_fig, ei_freq_info = plot_ei_frequency_analysis(
        network=network,
        activity_record=activity_record,
        dt=dt,
        figsize=(14, 8),
        dpi=150,
        save_path="ei_frequency_analysis.png",
        darkstyle=True
    )

    # Print E/I frequency results
    print(f"\nE/I Frequency Analysis Results:")
    print(f"  Excitatory peak: {ei_freq_info['exc_peak_frequency_hz']:.1f} Hz {'(Gamma)' if ei_freq_info['exc_is_gamma'] else ''}")
    print(f"  Inhibitory peak: {ei_freq_info['inh_peak_frequency_hz']:.1f} Hz {'(Gamma)' if ei_freq_info['inh_is_gamma'] else ''}")

    # Generate E/I synchrony analysis (unique neurons vs total spikes at multiple timescales)
    print("\nGenerating E/I synchrony analysis (multiscale)...")
    sync_fig, sync_info = plot_ei_synchrony_analysis(
        network=network,
        activity_record=activity_record,
        dt=dt,
        figsize=(14, 12),
        dpi=150,
        save_path="ei_synchrony_analysis.png",
        darkstyle=True
    )

    # Print synchrony results (using 10ms reference bin)
    print(f"\nE/I Synchrony Analysis Results (at {sync_info['reference_bin_ms']}ms bins):")
    print(f"  E mean burst index: {sync_info['exc_mean_burst_ref']:.2f}")
    print(f"  I mean burst index: {sync_info['inh_mean_burst_ref']:.2f}")
    print(f"  (burst_idx=1.0 → population sync, >1.0 → neurons firing multiple times)")

    # Generate 3D distance-weight visualization for a sample neuron
    print("\nGenerating 3D distance-weight visualization...")
    neuron_to_visualize = 42 if n_neurons > 42 else 0
    visualize_distance_weights_3d(
        network=network,
        neuron_idx=neuron_to_visualize,
        save_path="distance_weights_3d.html"
    )

    # Generate 3D I→E distance-weight visualization (from one inhibitory neuron)
    print("\nGenerating 3D I→E distance-weight visualization...")
    # Find an inhibitory neuron to visualize
    inhibitory_neuron_idx = None
    for i in range(n_neurons):
        if network.neurons[i].is_inhibitory:
            inhibitory_neuron_idx = i
            break
    visualize_ie_distance_weights_3d(
        network=network,
        neuron_idx=inhibitory_neuron_idx,
        save_path="ie_distance_weights_3d.html"
    )

    print("\n=== Experiment complete ===")

    return network, activity_record, neuron_data, stimulation_record


def run_biologically_plausible_simulation(random_seed=42):
    """
    Run a 3D spherical simulation with biologically plausible parameters.

    Stimulation Modes:
    ------------------
    - current_injection_stimulation=True: At each stim_interval, selected neurons
      receive sustained current for current_injection_duration timesteps.

    - poisson_process_stimulation=True (requires current_injection=False): At each
      stim_interval, a window opens where each selected neuron has poisson_process_probability
      probability of receiving stimulation per timestep, for poisson_process_duration timesteps.

    Parameters:
    -----------
    random_seed : int
        Random seed for reproducibility
    """
    return run_spherical_experiment(
        # Network parameters (same as circular experiment)
        n_neurons=6000,
        connection_p=0.1,  # Fallback default
        connection_probabilities={  # Biological connection probabilities
            'ee': 0.10,   # E→E: ~10% (local recurrent excitation)
            'ei': 0.15,   # E→I: ~15-20% (feedforward to interneurons)
            'ie': 0.35,   # I→E: ~40-50% (strong blanket inhibition)
            'ii': 0.15,   # I→I: ~10-20% (interneuron networks)
        },
        weight_scale=0.3,
        inhibitory_fraction=0.20,
        transmission_delay=1,
        distance_lambda=0.15, # Distance decay for all connection types (except I→E)
        lambda_decay_ie=0.08,  # Slower decay for I→E connections (longer range inhibition)

        # Simulation parameters
        duration=500, #ms
        dt=0.1,

        # Stimulation parameters
        stim_interval=100, # the interval beteeen stims
        stim_interval_strength=10, # strength of each stim
        stim_fraction=0.20, # fraction of total neurons to stimulate each interval
        no_stimulation=False,
        stochastic_stim=False,

        # Stimulation mode parameters
        current_injection_stimulation=False, # Sustained current for N timesteps
        current_injection_duration=200, # How many timesteps current is applied
        poisson_process_stimulation=True,  # Probabilistic stim (only if current_injection=False)
        poisson_process_probability=0.005, #per timestep probability of stim in stim_fraction neurons
        poisson_process_duration=50,

        # Noise parameters
        enable_noise=True,
        v_noise_amp=0.6,
        i_noise_amp=0.003,

        # Reversal potential parameters
        e_reversal=0.0,
        i_reversal=-80.0,

        # Other parameters
        layout='sphere',
        random_seed=random_seed,
        animate=True,
        max_animation_frames=5000,

        # Neuron parameter jitter
        # Voltage params: Gaussian (std dev in mV)
        jitter_v_rest=3.0,              # ±3mV Gaussian jitter on resting potential
        jitter_v_threshold=2.5,         # ±2.5mV Gaussian jitter on threshold
        # Time constants: Log-normal (coefficient of variation, CV = σ/μ)
        jitter_tau_m=0.3,               # 30% CV on membrane time constant
        jitter_tau_ref=0.25,            # 25% CV on refractory period
        jitter_tau_e=0.3,               # 30% CV on excitatory synaptic tau
        jitter_tau_i=0.3,               # 30% CV on inhibitory synaptic tau
        jitter_adaptation_increment=0.4, # 40% CV on adaptation increment
        jitter_tau_adaptation=0.35      # 35% CV on adaptation time constant
    )


if __name__ == "__main__":
    # Run the 3D spherical experiment with biologically plausible parameters
    network, activity_record, neuron_data, stimulation_record = run_biologically_plausible_simulation()
