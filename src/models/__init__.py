import importlib
import os
import pkgutil

# 自动发现并导入当前目录下所有的 .py 模块，从而触发 @ModelRegistry.register
pkg_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([pkg_dir]):
    if module_name not in ["base", "model"]:  # 排除基类和占位类
        importlib.import_module(f"src.models.{module_name}")
