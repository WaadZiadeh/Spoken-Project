Can traditional speech/signal-processing features capture acoustic artifacts that allow us to distinguish human speech from AI-generated/deepfake speech?

                    AUDIO
                      │
             ┌────────┴────────┐
             │                 │
          REAL             AI-GENERATED
             │                 │
             └────────┬────────┘
                      ↓
                PREPROCESSING
                      ↓
               FRAME THE AUDIO
                      ↓
             FEATURE EXTRACTION
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
     MFCC            LFCC       SPECTRAL
       │              │              │
       └──────────────┼──────────────┘
                      ↓
              FEATURE COMBINATION
                      ↓
               ML CLASSIFIER
                      ↓
              ┌───────┴───────┐
              ↓               ↓
            REAL         AI-GENERATED
                      ↓
                 EVALUATION
                      ↓
 Accuracy / Precision / Recall
 F1 / Confusion Matrix / ROC-AUC
                      ↓
                COMPARISON
                      ↓
              GENERALIZATION


 Training set
     ↓
Train model

Validation set
     ↓
Choose/tune model

Test set
     ↓
Final evaluation
