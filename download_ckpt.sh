#!/bin/bash

# 设置下载根目录, 默认为当前目录
DOWNLOAD_DIR="ckpt/"
mkdir -p "$DOWNLOAD_DIR"  # 如果目录不存在，则创建它

# 定义多个 URL
URLs=(
  "https://huggingface.co/novateur/jingyu/resolve/main/ibq_epoch63.ckpt?download=true"
  "https://huggingface.co/novateur/jingyu/resolve/main/mimi_decoder_ckpt_epoch113.ckpt?download=true"
  "https://huggingface.co/novateur/jingyu/resolve/main/mimi_decoder.zip?download=true"
  # 可以继续添加更多的 URL
)

# 循环遍历每个 URL
for URL in "${URLs[@]}"; do
  # 从 URL 中提取文件名
  FILE_NAME=$(basename "$URL" | sed 's/?download=true//')

  # 设置完整的下载路径
  FILE_PATH="$DOWNLOAD_DIR/$FILE_NAME"

  # 使用 wget 下载文件（启用断点续传）
  echo "正在下载：$FILE_NAME 到 $FILE_PATH"
  wget -c -O "$FILE_PATH" "$URL"

  # 打印下载完成的信息
  echo "$FILE_NAME 下载完成，保存路径：$FILE_PATH"
done
