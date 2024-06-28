import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pathlib import Path
import glob

# ローカルのディレクトリパスに変更
train_dir = Path("C:/Users/YourName/Documents/data/2回目/train")

# 指定したパターンに一致するすべてのファイルパスのリストを取得
file_paths = glob.glob("C:/Users/YourName/Documents/data/2回目/*/*/*.jpg")

# ファイルの総数を計算
file_count = len(file_paths)

# ファイルの総数を表示
print(f"ファイルの総数: {file_count}")

# パラメータ設定
batch_size = 32
img_height = 285
img_width = 162

# トレーニングデータセットを読み込む
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# バリデーションデータセットを読み込む
val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# クラス名の命名
class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# 基本的な Keras モデル
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# モデルをトレーニングする
epochs = 100  # N回
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# トレーニング精度とバリデーション精度を取得
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# エポックの範囲を定義
epochs_range = range(epochs)

# トレーニングの結果を視覚化する
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# モデルの保存
model.save('retrained_model2')
with open('retrained_labels.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
