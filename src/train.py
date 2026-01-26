# def train_cnn_pipeline():
#     import os
#     import cv2
#     import numpy as np
#     import pandas as pd
#     import warnings
#     import matplotlib.pyplot as plt

#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import LabelEncoder

#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import (
#         Conv2D, MaxPooling2D, Flatten,
#         Dense, Dropout, BatchNormalization
#     )
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.preprocessing.image import ImageDataGenerator

#     # =================================================
#     # SUPPRESS WARNINGS
#     # =================================================
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#     warnings.filterwarnings("ignore")

#     # =================================================
#     # PATHS (UPDATED PROJECT NAME)
#     # =================================================
#     BASE_DIR = r"D:\INSURENCEFRAUDDETECTION"
#     CSV_PATH = os.path.join(
#         BASE_DIR,
#         "dataset", "processed", "labels_clean_final.csv"
#     )
#     IMAGE_ROOT = os.path.join(
#         BASE_DIR,
#         "dataset", "raw", "vehicle_damage_dataset"
#     )

#     # =================================================
#     # PARAMETERS
#     # =================================================
#     IMG_SIZE = (224, 224)
#     BATCH_SIZE = 32
#     EPOCHS = 10
#     RANDOM_STATE = 42

#     TRAIN_SIZE = 0.70
#     VAL_SIZE   = 0.15
#     TEST_SIZE  = 0.15

#     # =================================================
#     # LOAD CSV
#     # =================================================
#     df = pd.read_csv(CSV_PATH)
#     print("✅ CSV Loaded:", df.shape)

#     # Add class column if missing
#     if "class" not in df.columns:
#         df["class"] = df["image_path"].apply(
#             lambda x: "real" if x.lower().startswith("real") else "fake"
#         )

#     # Encode labels
#     le = LabelEncoder()
#     df["label"] = le.fit_transform(df["class"])

#     print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

#     # =================================================
#     # TRAIN / VAL / TEST SPLIT (SINGLE SPLIT)
#     # =================================================
#     train_df, temp_df = train_test_split(
#         df,
#         test_size=(1 - TRAIN_SIZE),
#         stratify=df["label"],
#         random_state=RANDOM_STATE
#     )

#     val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

#     val_df, test_df = train_test_split(
#         temp_df,
#         test_size=(1 - val_ratio),
#         stratify=temp_df["label"],
#         random_state=RANDOM_STATE
#     )

#     # =================================================
#     # SPLIT SUMMARY (IMPORTANT FOR PROJECT)
#     # =================================================
#     total = len(df)
#     print("\n✅ DATASET SPLIT SUMMARY")
#     print(f"Total Samples : {total}")
#     print(f"Train         : {len(train_df)} ({len(train_df)/total*100:.2f}%)")
#     print(f"Validation    : {len(val_df)} ({len(val_df)/total*100:.2f}%)")
#     print(f"Test          : {len(test_df)} ({len(test_df)/total*100:.2f}%)")

#     # =================================================
#     # IMAGE DATA GENERATORS
#     # =================================================
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=15,
#         zoom_range=0.1,
#         horizontal_flip=True
#     )

#     val_test_datagen = ImageDataGenerator(rescale=1./255)

#     train_gen = train_datagen.flow_from_dataframe(
#         train_df,
#         directory=IMAGE_ROOT,
#         x_col="image_path",
#         y_col="class",
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="binary",
#         shuffle=True
#     )

#     val_gen = val_test_datagen.flow_from_dataframe(
#         val_df,
#         directory=IMAGE_ROOT,
#         x_col="image_path",
#         y_col="class",
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="binary",
#         shuffle=False
#     )

#     test_gen = val_test_datagen.flow_from_dataframe(
#         test_df,
#         directory=IMAGE_ROOT,
#         x_col="image_path",
#         y_col="class",
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="binary",
#         shuffle=False
#     )

#     # =================================================
#     # BUILD CNN MODEL
#     # =================================================
#     def build_cnn_model(input_shape=(224,224,3)):
#         model = Sequential()

#         model.add(Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(2,2))

#         model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(2,2))

#         model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(2,2))

#         model.add(Flatten())
#         model.add(Dense(256, activation="relu"))
#         model.add(Dropout(0.5))
#         model.add(Dense(128, activation="relu"))
#         model.add(Dropout(0.3))

#         model.add(Dense(1, activation="sigmoid"))

#         model.compile(
#             optimizer=Adam(1e-4),
#             loss="binary_crossentropy",
#             metrics=["accuracy"]
#         )

#         return model

#     model = build_cnn_model()
#     model.summary()

#     # =================================================
#     # TRAIN MODEL
#     # =================================================
#     history = model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=EPOCHS,
#         verbose=2
#     )

#     # =================================================
#     # EVALUATE ON TEST SET
#     # =================================================
#     test_loss, test_acc = model.evaluate(test_gen, verbose=0)
#     print("\n📊 TEST SET PERFORMANCE")
#     print(f"Test Accuracy : {test_acc*100:.2f}%")
#     print(f"Test Loss     : {test_loss:.4f}")

#     # =================================================
#     # SAVE MODEL
#     # =================================================
#     os.makedirs("saved_models", exist_ok=True)
#     model.save("saved_models/final_cnn_model.h5")
#     print("\n✅ Model saved at saved_models/final_cnn_model.h5")
def train_cnn_pipeline():
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import warnings

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten,
        Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping

    # =================================================
    # SUPPRESS WARNINGS
    # =================================================
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")

    # =================================================
    # PATHS
    # =================================================
    BASE_DIR = r"D:\INSURENCEFRAUDDETECTION"
    CSV_PATH = os.path.join(BASE_DIR, "dataset", "processed", "labels_clean_final.csv")
    IMAGE_ROOT = os.path.join(BASE_DIR, "dataset", "raw", "vehicle_damage_dataset")

    # =================================================
    # PARAMETERS
    # =================================================
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    RANDOM_STATE = 42

    TRAIN_SIZE = 0.70
    VAL_SIZE   = 0.15
    TEST_SIZE  = 0.15

    # =================================================
    # LOAD CSV
    # =================================================
    df = pd.read_csv(CSV_PATH)
    print("✅ CSV Loaded:", df.shape)

    # Ensure class column exists
    if "class" not in df.columns:
        df["class"] = df["image_path"].apply(
            lambda x: "real" if x.lower().startswith("real") else "fake"
        )

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["class"])
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # =================================================
    # TRAIN / VAL / TEST SPLIT (SINGLE SPLIT)
    # =================================================
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_SIZE),
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    # =================================================
    # SPLIT SUMMARY
    # =================================================
    total = len(df)
    print("\n✅ DATASET SPLIT SUMMARY")
    print(f"Total Samples : {total}")
    print(f"Train         : {len(train_df)} ({len(train_df)/total*100:.2f}%)")
    print(f"Validation    : {len(val_df)} ({len(val_df)/total*100:.2f}%)")
    print(f"Test          : {len(test_df)} ({len(test_df)/total*100:.2f}%)")

    # =================================================
    # IMAGE DATA GENERATORS (STRONG AUGMENTATION)
    # =================================================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=IMAGE_ROOT,
        x_col="image_path",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_dataframe(
        val_df,
        directory=IMAGE_ROOT,
        x_col="image_path",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_dataframe(
        test_df,
        directory=IMAGE_ROOT,
        x_col="image_path",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # =================================================
    # BUILD CNN MODEL (REDUCED COMPLEXITY)
    # =================================================
    def build_cnn_model(input_shape=(224,224,3)):
        model = Sequential()

        model.add(Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))

        model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))

        model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))   # 🔽 reduced
        model.add(Dropout(0.6))                    # 🔼 increased

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    model = build_cnn_model()
    model.summary()

    # =================================================
    # EARLY STOPPING (KEY OVERFITTING FIX)
    # =================================================
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # =================================================
    # TRAIN MODEL
    # =================================================
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=2
    )

    # =================================================
    # EVALUATE ON TEST SET
    # =================================================
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print("\n📊 TEST SET PERFORMANCE")
    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    # =================================================
    # SAVE MODEL
    # =================================================
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/final_cnn_model.h5")
    print("\n✅ Model saved at saved_models/final_cnn_model.h5")

# =================================================
# RUN PIPELINE
# =================================================
if __name__ == "__main__":
    train_cnn_pipeline()
