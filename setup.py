from setuptools import setup, find_packages

with open("version" , "r") as f:
    _version = f.readline()
setup(
    name="lyy19Lib",
    version=_version,
    author="lyy19",
    author_email="lyy2286301015@126.com",
    description="各种类及函数",  # 库的简要描述
    long_description=open("README.md", encoding="utf-8").read(),  # 读取README作为长描述
    long_description_content_type="text/markdown",
    url="https://github.com/lyy1119/PythonLib",
    packages=find_packages("./lyy19Lib"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # 最低 Python 版本要求
    install_requires=[
        # 这里填写依赖项，例如：
        # "numpy>=1.19.0",
    ],
)
