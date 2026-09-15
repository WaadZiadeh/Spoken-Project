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




## Waad: MFCC + SVM experiment

See [MFCC implementation and Colab instructions](code/mfcc/README.md). This separate
experiment uses the existing train/validation/test CSVs, caches 40-dimensional
MFCC features, selects SVM parameters on validation F1, and saves test metrics
under `results/mfcc_svm/`. Dataset audio stays in Drive; the LFCC notebook is unchanged.
