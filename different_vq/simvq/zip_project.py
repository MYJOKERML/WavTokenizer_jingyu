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
            if rel_root == ".":
                rel_root = ""

            # 排除 .git 目录
            if '.git' in dirs:
                dirs.remove('.git')
            
            # 动态排除被 .gitignore 忽略的目录
            dirs_to_remove = []
            for d in dirs:
                dir_rel_path = os.path.normpath(os.path.join(rel_root, d))
                # 在路径末尾添加 '/' 以明确这是一个目录
                if combined_ignore(dir_rel_path + "/"):
                    dirs_to_remove.append(d)
            
            # 从 dirs 中移除被忽略的目录，防止 os.walk 进入这些目录
            for d in dirs_to_remove:
                dirs.remove(d)
                print(f'Ignoring directory: {os.path.join(rel_root, d)}')

            for file in files:
                filepath = os.path.join(root, file)
                relpath = os.path.normpath(os.path.join(rel_root, file))
                # 检查是否被忽略
                if combined_ignore(relpath):
                    print(f'Ignoring file: {relpath}')
                    continue
                print(f'Zipping {relpath}')
                # 添加文件到 Zip，使用相对路径
                zipf.write(filepath, relpath)
    
    print(f"压缩完成：{archive_name}")

if __name__ == "__main__":
    archive_name = 'replace_simvq.zip'
    create_zip_archive(archive_name=archive_name)
