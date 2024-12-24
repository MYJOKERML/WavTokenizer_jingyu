import os
import zipfile
from gitignore_parser import parse_gitignore

def create_zip_archive(archive_name='archive.zip', base_dir='.'):
    # 查找所有 .gitignore 文件
    gitignore_files = []
    for root, dirs, files in os.walk(base_dir):
        if '.git' in dirs:
            dirs.remove('.git')  # 排除 .git 目录
        for file in files:
            if file == '.gitignore':
                gitignore_files.append(os.path.join(root, file))
    
    # 生成忽略函数
    # 优先级较高的 .gitignore 覆盖优先级较低的
    ignore_funcs = [parse_gitignore(gitignore) for gitignore in gitignore_files]
    
    def combined_ignore(path):
        for ignore in ignore_funcs:
            if ignore(path):
                return True
        return False
    
    # 创建 Zip 文件
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # 计算相对路径
            rel_root = os.path.relpath(root, base_dir)
            
            # 排除 .git 目录
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                filepath = os.path.join(root, file)
                relpath = os.path.relpath(filepath, base_dir)
                
                # 检查是否被忽略
                if combined_ignore(relpath):
                    continue
                
                # 添加文件到 Zip，使用相对路径
                zipf.write(filepath, relpath)
    
    print(f"压缩完成：{archive_name}")

if __name__ == "__main__":
    archive_name = 'mimi_transformer_attn.zip'
    create_zip_archive(archive_name=archive_name)
