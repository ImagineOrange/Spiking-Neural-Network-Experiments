%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#333333', 'primaryTextColor': '#fff', 'primaryBorderColor': '#7C0000', 'lineColor': '#F8B229', 'secondaryColor': '#006100', 'tertiaryColor': '#444444', 'fontFamily': 'Arial' }}}%%
flowchart TB
    %% Node styles with rounded corners
    A(["MNIST Example\n28x28 pixels"]) --> B{{"Encoding Mode?"}}
    B -->|"intensity_to_neuron"| C(["Downsample Image\n28x28 → 7x7\nFactor=4"])
    B -->|"conv_feature_to_neuron"| D(["Full Resolution Image\n28x28"])
    
    D --> E(["ConvNet Layer 1\n16 channels, 14x14"])
    E --> F(["ConvNet Layer 2\n32 channels, 7x7"])
    F --> G(["ConvNet Layer 3\n1 channel, 7x7"])
    G --> H(["Flatten to\n49 Features"])
    
    C --> I(["Rate Code Encoding\nPixel Intensity to Firing Rate"])
    I --> K(["Input Spike Trains\nfor 49 Pixels (7x7)"])
    
    H --> J(["Rate Code Encoding\nFeature Value to Firing Rate"])
    J --> L(["Input Spike Trains\nfor 49 CNN Features"])
    
    K --> M(["Input Layer\n49 Neurons"])
    L --> M
    M --> N(["Hidden Layer 1\n30 Neurons"])
    N --> O(["Hidden Layer 2\n20 Neurons"])
    O --> P(["Output Layer\nN_CLASSES Neurons"])
    
    Q(["Chromosome\nConnection Weights"]) --> R(["Fitness Evaluation\nClassification Accuracy"])
    R --> S(["Selection, Crossover\nMutation"]) 
    S --> T(["Best Weights"])
    T --> Q
    T -.-> M
    
    %% Rounded subgraph definitions
    subgraph InputProcessing["Input Processing"]
        A
        B
        C
        D
    end
    
    subgraph ConvNetModule["ConvNet Feature Extraction"]
        E
        F
        G
        H
    end
    
    subgraph SpikeEncoding["Spike Encoding"]
        I
        J
        K
        L
    end
    
    subgraph SNNArchitecture["SNN Architecture"]
        M
        N
        O
        P
    end
    
    subgraph GeneticAlgorithm["Genetic Algorithm"]
        Q
        R
        S
        T
    end
    
    %% Color definitions
    classDef inputClass fill:#FF9999,stroke:#FF0000,stroke-width:2px,color:black
    classDef convClass fill:#99CCFF,stroke:#0066CC,stroke-width:2px,color:black
    classDef spikeClass fill:#FFCC99,stroke:#FF9900,stroke-width:2px,color:black
    classDef snnClass fill:#99FF99,stroke:#33CC33,stroke-width:2px,color:black
    classDef gaClass fill:#CC99FF,stroke:#9933FF,stroke-width:2px,color:black
    
    %% Subgraph styling for rounded corners
    classDef subgraphStyle fill:#333333,color:white,stroke:#999999,stroke-width:2px,rx:10,ry:10
    
    class InputProcessing,ConvNetModule,SpikeEncoding,SNNArchitecture,GeneticAlgorithm subgraphStyle
    
    %% Apply colors
    class A,B,C,D inputClass
    class E,F,G,H convClass
    class I,J,K,L spikeClass
    class M,N,O,P snnClass
    class Q,R,S,T gaClass