'''
Created on 15.12.2014

@author: Peter U. Diehl

Modified for 4:1 Excitatory:Inhibitory Ratio Experiment with Local Inhibition

Key changes from original 1:1 architecture (v2 - conservative overlap reduction):
- E->I connectivity: 5% sparse (reduced from 12% to lower inhibition overlap)
- I->E connectivity: 30% sparse (reduced from 90% to create local competition)
- I->E weight: 10.0 (calibrated to match original drive)

Rationale for v2 changes:
The v1 4:1 config (12% E->I, 90% I->E, weight=3.0) achieved 82.78% accuracy but
crushed digit "1" (only 6 neurons assigned). The problem was ~100% inhibition overlap,
not drive strength (v1 drive was 32.4, well above original's 17.0).

v2 makes a conservative adjustment: reduce overlap from 100% to 83% while keeping
drive close to original (15.0 vs 17.0). This should allow weak stimuli to compete
without fundamentally changing the network dynamics.

Inhibitory Drive Calibration (v2):
- Original 1:1: 1 E spike -> 1 I neuron -> 17.0 inhibition to 399 E neurons
- v1 4:1: 1 E spike -> 12 I neurons -> 10.8 expected hits @ 3.0 = 32.4 drive, 100% overlap
- v2 4:1: 1 E spike -> 5 I neurons -> 1.5 expected hits @ 10.0 = 15.0 drive, 83% overlap

References:
- Packer & Yuste (2011). Dense, Unspecific Connectivity of Neocortical PV+ Interneurons.
- Bhattacharjee et al. (2022). Local Connections of Pyramidal Neurons to PV Interneurons.
- Tremblay et al. (2016). GABAergic Interneurons in the Neocortex (E:I ratios).
'''

import scipy.ndimage as sp
import numpy as np
import pylab
import os
from config import Config


def randomDelay(minDelay, maxDelay):
    return np.random.rand()*(maxDelay-minDelay) + minDelay


def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos


def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList


def create_weights():
    # Load network architecture from config
    cfg = Config()
    nInput = cfg.n_input
    nE = cfg.n_e
    nI = cfg.n_i
    dataPath = cfg.data_path + 'random/'
    os.makedirs(dataPath, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Creating random connection weights for 4:1 E:I architecture")
    print(f"{'='*60}")
    print(f"Network dimensions:")
    print(f"  Input neurons:      {nInput}")
    print(f"  Excitatory neurons: {nE}")
    print(f"  Inhibitory neurons: {nI}")
    print(f"  E:I ratio:          {nE}:{nI} = {nE/nI:.1f}:1")
    print(f"{'='*60}\n")

    weight = {}
    weight['ee_input'] = 0.3
    weight['ei_input'] = 0.2
    weight['ee'] = 0.1
    weight['ei'] = 10.4
    weight['ie'] = 11.3        # Calibrated: 5 I neurons × 0.30 × 11.3 = 17.0 drive (matches original)
    weight['ii'] = 0.4

    pConn = {}
    pConn['ee_input'] = 1.0    # Input -> Excitatory: 100% (all-to-all)
    pConn['ei_input'] = 0.1    # Input -> Inhibitory: 10% (as specified in paper)
    pConn['ee'] = 1.0          # Excitatory -> Excitatory: not used
    pConn['ei'] = 0.05         # Excitatory -> Inhibitory: 5% (reduced from 12% to lower overlap)
    pConn['ie'] = 0.30         # Inhibitory -> Excitatory: 30% (reduced from 90% for ~83% overlap)
    pConn['ii'] = 0.1          # Inhibitory -> Inhibitory: not used


    print('Creating Input -> Excitatory connections (XeAe)')
    print(f'  Connectivity: {pConn["ee_input"]*100:.0f}% (all-to-all)')
    print(f'  Weight range: ~{weight["ee_input"]} (will be normalized to sum={cfg.weight_ee_input})')
    print(f'  Expected connections: {nInput * nE} ({nInput}x{nE})')
    connNameList = ['XeAe']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nE)) + 0.01
        weightMatrix *= weight['ee_input']
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = [(i, j, weightMatrix[i,j]) for j in range(nE) for i in range(nInput)]
        np.save(dataPath+name, weightList)
        print(f'  ✓ Saved {len(weightList)} connections to {dataPath}{name}.npy')


    print('\nCreating Input -> Inhibitory connections (XeAi)')
    print(f'  Connectivity: {pConn["ei_input"]*100:.0f}% (sparse, as specified in paper)')
    print(f'  Weight: {weight["ei_input"]}')
    print(f'  Expected connections: ~{int(nInput * nI * pConn["ei_input"])} ({nInput}x{nI}x{pConn["ei_input"]})')
    connNameList = ['XeAi']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nI))
        weightMatrix *= weight['ei_input']
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei_input'])
        np.save(dataPath+name, weightList)
        print(f'  ✓ Saved {len(weightList)} connections to {dataPath}{name}.npy')


    print('\nCreating Excitatory -> Inhibitory connections (AeAi)')
    print(f'  Architecture: 4:1 ratio (nE={nE}, nI={nI})')
    print(f'  Connectivity: {pConn["ei"]*100:.0f}% (sparse for local competition)')
    print(f'  Weight: {weight["ei"]}')
    print(f'  Expected connections: ~{int(nE * nI * pConn["ei"])} ({nE}x{nI}x{pConn["ei"]})')
    print(f'  I neurons per E spike: ~{int(nI * pConn["ei"])}')
    print(f'  E inputs per I neuron: ~{int(nE * pConn["ei"])}')
    print(f'  Note: Reduced from 12% to 5% to lower inhibition overlap')
    connNameList = ['AeAi']
    for name in connNameList:
        # For 4:1 case (nE != nI), use sparse connectivity for local competition
        # Each excitatory neuron connects to ~5% of inhibitory neurons (~5 I per E)
        weightMatrix = np.random.random((nE, nI))
        weightMatrix *= weight['ei']
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        np.save(dataPath+name, weightList)
        print(f'  ✓ Saved {len(weightList)} connections to {dataPath}{name}.npy')


    print('\nCreating Inhibitory -> Excitatory connections (AiAe)')
    print(f'  Architecture: Rectangular matrix ({nI}x{nE})')
    print(f'  Connectivity: {pConn["ie"]*100:.0f}% (sparse for local inhibition)')
    print(f'  Weight: {weight["ie"]} (calibrated for ~15 drive)')
    print(f'  Expected connections: ~{int(nI * nE * pConn["ie"])} ({nI}x{nE}x{pConn["ie"]})')
    print(f'  Note: Each inhibitory neuron inhibits ~{int(nE * pConn["ie"])} excitatory neurons')
    print(f'  Note: Reduced from 90% to 30% to create ~83% overlap (vs 100%)')

    # Calculate and display inhibitory drive comparison
    i_per_e_spike = nI * pConn['ei']
    inhib_drive = i_per_e_spike * pConn['ie'] * weight['ie']
    print(f'\n  Inhibitory Drive Calculation:')
    print(f'    1 E spike -> {i_per_e_spike:.0f} I neurons activated')
    print(f'    Each I neuron -> inhibits {pConn["ie"]*100:.0f}% of E neurons')
    print(f'    Inhibition per target E = {i_per_e_spike:.0f} × {pConn["ie"]} × {weight["ie"]} = {inhib_drive:.1f}')
    print(f'    Original 1:1 inhibition = 17.0 (ratio: {inhib_drive/17.0:.2f}x)')

    connNameList = ['AiAe']
    for name in connNameList:
        # Sparse random connectivity provides local inhibition
        # Each inhibitory neuron connects to ~30% of excitatory neurons
        weightMatrix = np.random.random((nI, nE))
        weightMatrix *= weight['ie']
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        np.save(dataPath+name, weightList)
        print(f'  ✓ Saved {len(weightList)} connections to {dataPath}{name}.npy')

    print(f"\n{'='*60}")
    print("Weight generation complete!")
    print(f"{'='*60}")
    print("Summary of connection matrices:")
    print(f"  XeAe: {nInput}x{nE} @ {pConn['ee_input']*100:.0f}% = {nInput*nE} connections")
    print(f"  XeAi: {nInput}x{nI} @ {pConn['ei_input']*100:.0f}% = ~{int(nInput*nI*pConn['ei_input'])} connections")
    print(f"  AeAi: {nE}x{nI} @ {pConn['ei']*100:.0f}% = ~{int(nE*nI*pConn['ei'])} connections")
    print(f"  AiAe: {nI}x{nE} @ {pConn['ie']*100:.0f}% = ~{int(nI*nE*pConn['ie'])} connections")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    create_weights()
