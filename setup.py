"""
医疗推荐系统安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-recommendation-system",
    version="0.1.0",
    author="大创项目团队",
    author_email="your-email@example.com",
    description="基于知识图谱和向量检索的智能医疗推荐系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/medical-recommendation-system",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "full": [
            "langchain>=0.1.0",
            "openai>=1.6.0",
            "chromadb>=0.4.22",
            "sentence-transformers>=2.2.2",
            "neo4j>=5.14.0",
            "fastapi>=0.108.0",
            "streamlit>=1.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-recommend=scripts.run_agent_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "medical_recommendation": ["config/*.yaml"],
    },
)