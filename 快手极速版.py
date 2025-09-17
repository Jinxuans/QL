#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeaderKS - SO文件管理和加载器
优化的SO文件下载、更新和加载系统

新功能：自动依赖检测和安装
- 自动检测SO模块加载时缺失的Python依赖
- 自动安装常见依赖包（如aiohttp-socks等）
- 支持通过环境变量控制自动安装行为
- 提供详细的错误信息和安装建议

环境变量配置：
- LEADERKS_SERVER_URL=url                # 自定义服务器地址（可选）

使用示例：
1. 基本使用（自动安装依赖和更新）：
   python LeaderKS1.0.py

2. 自定义服务器地址：
   export LEADERKS_SERVER_URL=http://your-server.com:port
   python LeaderKS1.0.py

3. 程序化使用：
   from LeaderKS1.0 import LeaderKS, ServerConfig, UpdateConfig
   
   config = ServerConfig()
   update_config = UpdateConfig()  # 默认开启所有功能
   leader_ks = LeaderKS(config, update_config)
   exit_code = leader_ks.run("Kuaishou")
"""

import platform
import sys
import os
import subprocess
import shutil
import importlib.util
import requests
import hashlib
import json
import time
import asyncio
import logging
import functools
import re
import pkg_resources
from typing import Optional, Tuple, Dict, Any, Union, Callable, List
from urllib.parse import urljoin
from dataclasses import dataclass
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('leaderks.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def performance_monitor(func: Callable) -> Callable:
    """性能监控装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"{func.__name__} 执行耗时: {elapsed_time:.3f} 秒")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败，耗时: {elapsed_time:.3f} 秒，错误: {e}")
            raise
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 所有尝试都失败了")
            raise last_exception
        return wrapper
    return decorator

@dataclass
class ServerConfig:
    """服务器配置"""
    base_url: str = 'http://154.12.60.33:2424'
    download_endpoint: str = '/api/download_so.php'
    check_update_endpoint: str = '/api/check_update.php'
    timeout: int = 30
    retry_times: int = 3
    chunk_size: int = 8192
    retry_delay: int = 2

@dataclass
class UpdateConfig:
    """更新配置"""
    auto_update: bool = True
    ask_confirmation: bool = False
    backup_old_files: bool = True
    delete_backup_after_success: bool = True
    auto_install_dependencies: bool = True  # 自动安装缺失依赖

@dataclass
class SystemInfo:
    """系统信息"""
    architecture: str
    python_version_tag: str
    platform_info: str
    python_version: str

class DependencyManager:
    """依赖管理类"""
    
    # 常见依赖包映射 - 从错误信息到包名的映射
    DEPENDENCY_MAPPING = {
        'aiohttp_socks': 'aiohttp-socks',
        'aiohttp_socks_proxy': 'aiohttp-socks',
        'aiohttp_proxy': 'aiohttp-proxy',
        'aiohttp': 'aiohttp',
        'asyncio': 'asyncio',  # 通常内置
        'requests': 'requests',
        'urllib3': 'urllib3',
        'certifi': 'certifi',
        'charset_normalizer': 'charset-normalizer',
        'idna': 'idna',
        'cryptography': 'cryptography',
        'pycryptodome': 'pycryptodome',
        'pycryptodomex': 'pycryptodomex',
        'lxml': 'lxml',
        'beautifulsoup4': 'beautifulsoup4',
        'selenium': 'selenium',
        'webdriver_manager': 'webdriver-manager',
        'fake_useragent': 'fake-useragent',
        'user_agents': 'user-agents',
        'pytz': 'pytz',
        'dateutil': 'python-dateutil',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'transformers': 'transformers',
        'openai': 'openai',
        'anthropic': 'anthropic',
        'google': 'google-cloud',
        'boto3': 'boto3',
        'azure': 'azure-sdk',
        'redis': 'redis',
        'pymongo': 'pymongo',
        'sqlalchemy': 'sqlalchemy',
        'psycopg2': 'psycopg2-binary',
        'mysql': 'mysql-connector-python',
        'sqlite3': 'sqlite3',  # 通常内置
        'json': 'json',  # 内置
        'base64': 'base64',  # 内置
        'hashlib': 'hashlib',  # 内置
        'hmac': 'hmac',  # 内置
        'uuid': 'uuid',  # 内置
        'datetime': 'datetime',  # 内置
        'time': 'time',  # 内置
        'os': 'os',  # 内置
        'sys': 'sys',  # 内置
        're': 're',  # 内置
        'math': 'math',  # 内置
        'random': 'random',  # 内置
        'collections': 'collections',  # 内置
        'itertools': 'itertools',  # 内置
        'functools': 'functools',  # 内置
        'operator': 'operator',  # 内置
        'copy': 'copy',  # 内置
        'pickle': 'pickle',  # 内置
        'shelve': 'shelve',  # 内置
        'dbm': 'dbm',  # 内置
        'zlib': 'zlib',  # 内置
        'gzip': 'gzip',  # 内置
        'bz2': 'bz2',  # 内置
        'lzma': 'lzma',  # 内置
        'zipfile': 'zipfile',  # 内置
        'tarfile': 'tarfile',  # 内置
        'csv': 'csv',  # 内置
        'configparser': 'configparser',  # 内置
        'argparse': 'argparse',  # 内置
        'getopt': 'getopt',  # 内置
        'logging': 'logging',  # 内置
        'warnings': 'warnings',  # 内置
        'contextlib': 'contextlib',  # 内置
        'abc': 'abc',  # 内置
        'atexit': 'atexit',  # 内置
        'traceback': 'traceback',  # 内置
        'gc': 'gc',  # 内置
        'inspect': 'inspect',  # 内置
        'site': 'site',  # 内置
        'sysconfig': 'sysconfig',  # 内置
        'platform': 'platform',  # 内置
        'subprocess': 'subprocess',  # 内置
        'threading': 'threading',  # 内置
        'multiprocessing': 'multiprocessing',  # 内置
        'concurrent': 'concurrent',  # 内置
        'queue': 'queue',  # 内置
        'sched': 'sched',  # 内置
        'socket': 'socket',  # 内置
        'ssl': 'ssl',  # 内置
        'select': 'select',  # 内置
        'selectors': 'selectors',  # 内置
        'signal': 'signal',  # 内置
        'mmap': 'mmap',  # 内置
        'ctypes': 'ctypes',  # 内置
        'struct': 'struct',  # 内置
        'codecs': 'codecs',  # 内置
        'unicodedata': 'unicodedata',  # 内置
        'stringprep': 'stringprep',  # 内置
        'readline': 'readline',  # 内置
        'rlcompleter': 'rlcompleter',  # 内置
        'cmd': 'cmd',  # 内置
        'shlex': 'shlex',  # 内置
        'tkinter': 'tkinter',  # 内置
        'turtle': 'turtle',  # 内置
        'pdb': 'pdb',  # 内置
        'profile': 'profile',  # 内置
        'pstats': 'pstats',  # 内置
        'timeit': 'timeit',  # 内置
        'trace': 'trace',  # 内置
        'faulthandler': 'faulthandler',  # 内置
        'tracemalloc': 'tracemalloc',  # 内置
        'distutils': 'distutils',  # 内置
        'ensurepip': 'ensurepip',  # 内置
        'venv': 'venv',  # 内置
        'zipapp': 'zipapp',  # 内置
        'runpy': 'runpy',  # 内置
        'importlib': 'importlib',  # 内置
        'pkgutil': 'pkgutil',  # 内置
        'modulefinder': 'modulefinder',  # 内置
        'runpy': 'runpy',  # 内置
        'pkg_resources': 'setuptools',
        'setuptools': 'setuptools',
        'pip': 'pip',
        'wheel': 'wheel',
    }
    
    def __init__(self):
        self.installed_packages = self._get_installed_packages()
    
    def _get_installed_packages(self) -> set:
        """获取已安装的包列表"""
        try:
            installed_packages = {pkg.project_name.lower() for pkg in pkg_resources.working_set}
            return installed_packages
        except Exception as e:
            logger.warning(f"无法获取已安装包列表: {e}")
            return set()
    
    def extract_missing_dependency(self, error_message: str) -> Optional[str]:
        """从ImportError消息中提取缺失的依赖包名"""
        # 匹配 "No module named 'xxx'" 模式
        pattern = r"No module named ['\"]([^'\"]+)['\"]"
        match = re.search(pattern, error_message)
        
        if match:
            module_name = match.group(1)
            # 处理子模块，如 'aiohttp_socks.proxy' -> 'aiohttp_socks'
            if '.' in module_name:
                module_name = module_name.split('.')[0]
            return module_name
        
        return None
    
    def get_package_name(self, module_name: str) -> Optional[str]:
        """根据模块名获取对应的包名"""
        # 直接查找映射
        if module_name in self.DEPENDENCY_MAPPING:
            package_name = self.DEPENDENCY_MAPPING[module_name]
            # 跳过内置模块
            if package_name == module_name and module_name in sys.builtin_module_names:
                return None
            return package_name
        
        # 尝试一些常见的转换规则
        # 下划线转连字符
        if '_' in module_name:
            package_name = module_name.replace('_', '-')
            return package_name
        
        # 直接使用模块名
        return module_name
    
    def is_package_installed(self, package_name: str) -> bool:
        """检查包是否已安装"""
        return package_name.lower() in self.installed_packages
    
    def install_package(self, package_name: str) -> bool:
        """安装指定的包"""
        try:
            logger.info(f"正在安装依赖包: {package_name}")
            
            # 使用pip安装
            cmd = [sys.executable, '-m', 'pip', 'install', package_name, '--upgrade']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                logger.info(f"✓ 成功安装依赖包: {package_name}")
                # 更新已安装包列表
                self.installed_packages.add(package_name.lower())
                return True
            else:
                logger.error(f"✗ 安装依赖包失败: {package_name}")
                logger.error(f"错误信息: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ 安装依赖包超时: {package_name}")
            return False
        except Exception as e:
            logger.error(f"✗ 安装依赖包时发生错误: {e}")
            return False
    
    def auto_install_dependency(self, error_message: str) -> bool:
        """自动检测并安装缺失的依赖"""
        # 提取缺失的模块名
        module_name = self.extract_missing_dependency(error_message)
        if not module_name:
            logger.warning("无法从错误信息中提取模块名")
            return False
        
        # 获取对应的包名
        package_name = self.get_package_name(module_name)
        if not package_name:
            logger.warning(f"无法确定模块 '{module_name}' 对应的包名")
            return False
        
        # 检查是否已安装
        if self.is_package_installed(package_name):
            logger.info(f"✓ 依赖包已安装: {package_name}")
            return True
        
        # 尝试安装
        logger.info(f"检测到缺失依赖: {module_name} -> {package_name}")
        return self.install_package(package_name)
    
    def check_and_install_common_dependencies(self) -> bool:
        """检查并安装常见依赖"""
        common_deps = ['requests', 'aiohttp', 'aiohttp-socks']
        all_installed = True
        
        for dep in common_deps:
            if not self.is_package_installed(dep):
                logger.info(f"检查并安装常见依赖: {dep}")
                if not self.install_package(dep):
                    all_installed = False
        
        return all_installed
    
    def get_installation_help(self, module_name: str) -> str:
        """获取手动安装依赖的帮助信息"""
        package_name = self.get_package_name(module_name)
        if package_name:
            return f"请手动安装依赖: pip install {package_name}"
        else:
            return f"请检查模块 '{module_name}' 是否正确，或手动安装相关依赖"
    
    def suggest_alternative_packages(self, module_name: str) -> List[str]:
        """建议可能的替代包"""
        suggestions = []
        
        # 基于模块名建议可能的包
        if 'socks' in module_name.lower():
            suggestions.extend(['aiohttp-socks', 'requests[socks]', 'pysocks'])
        elif 'http' in module_name.lower():
            suggestions.extend(['aiohttp', 'requests', 'httpx'])
        elif 'proxy' in module_name.lower():
            suggestions.extend(['aiohttp-socks', 'requests[socks]', 'pysocks'])
        
        return suggestions

class FileManager:
    """文件管理类"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.version_file = self.base_dir / 'version.json'
    
    def get_version_info_path(self) -> Path:
        """获取版本信息文件路径"""
        return self.version_file
    
    def save_version_info(self, version_info: Dict[str, Any]) -> bool:
        """保存版本信息到本地文件"""
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, ensure_ascii=False, indent=2)
            # logger.info(f"版本信息已保存: {version_info}")
            return True
        except Exception as e:
            # logger.error(f"保存版本信息失败: {e}")
            return False
    
    def load_version_info(self) -> Optional[Dict[str, Any]]:
        """从本地文件加载版本信息"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载版本信息失败: {e}")
        return None
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """计算文件的MD5哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return None
    
    def backup_file(self, file_path: Path, suffix: str = '.backup') -> Optional[Path]:
        """备份文件"""
        try:
            backup_path = file_path.with_suffix(file_path.suffix + suffix)
            if backup_path.exists():
                backup_path.unlink()
            shutil.move(str(file_path), str(backup_path))
            logger.info(f"已备份文件: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"备份文件失败: {e}")
            return None
    
    def restore_file(self, backup_path: Path, original_path: Path) -> bool:
        """恢复文件"""
        try:
            shutil.move(str(backup_path), str(original_path))
            logger.info(f"已恢复文件: {original_path}")
            return True
        except Exception as e:
            logger.error(f"恢复文件失败: {e}")
            return False

class NetworkManager:
    """网络管理类"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LeaderKS/2.0'})
    
    @performance_monitor
    def check_server_update(self, base_name: str, py_ver_tag: str, 
                           arch: str, current_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """检查服务器是否有更新版本"""
        try:
            url = urljoin(self.config.base_url, self.config.check_update_endpoint)
            data = {
                'base_name': base_name,
                'python_version': py_ver_tag,
                'architecture': arch,
                'current_version': current_version,
                'platform': platform.platform()
            }
            
            response = self.session.post(url, json=data, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('success'):
                return result.get('data')
            else:
                logger.error(f"服务器返回错误: {result.get('message', '未知错误')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"检查更新时发生错误: {e}")
            return None
    
    def request_so_download(self, base_name: str, py_ver_tag: str, 
                           arch: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """向服务器请求SO文件下载链接"""
        try:
            url = urljoin(self.config.base_url, self.config.download_endpoint)
            data = {
                'base_name': base_name,
                'python_version': py_ver_tag,
                'architecture': arch,
                'platform': platform.platform(),
                'client_info': {
                    'python_version': sys.version,
                    'platform': platform.platform(),
                    'architecture': arch
                }
            }
            
            response = self.session.post(url, json=data, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('success'):
                download_info = result.get('data', {})
                download_url = download_info.get('download_url')
                version_info = download_info.get('version_info', {})
                
                if download_url:
                    return download_url, version_info
                else:
                    logger.error("服务器未提供下载链接")
                    return None, None
            else:
                logger.error(f"服务器返回错误: {result.get('message', '未知错误')}")
                return None, None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {e}")
            return None, None
        except Exception as e:
            logger.error(f"请求下载时发生错误: {e}")
            return None, None
    
    @performance_monitor
    def download_so_file(self, base_name: str, py_ver_tag: str, 
                        arch: str, download_url: str) -> Optional[str]:
        """从服务器下载SO文件"""
        logger.info("开始下载SO文件")
        
        # 修正下载URL
        if download_url.startswith('http://154.12.60.33/') and ':2424' not in download_url:
            download_url = download_url.replace('http://154.12.60.33/', 'http://154.12.60.33:2424/')
            logger.info(f"修正后的下载地址: {download_url}")
        
        expected_filename = f"{base_name}.cpython-{py_ver_tag}-{arch}-linux-gnu.so"
        temp_filename = f"{expected_filename}.tmp"
        
        for attempt in range(self.config.retry_times):
            try:
                logger.info(f"下载尝试 {attempt + 1}/{self.config.retry_times}")
                
                response = self.session.get(
                    download_url, 
                    stream=True, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(temp_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
                    
                    f.flush()
                    os.fsync(f.fileno())
                
                print(f"\n下载完成: {downloaded_size} 字节")
                
                # 验证文件完整性
                if 'content-md5' in response.headers:
                    expected_hash = response.headers['content-md5']
                    actual_hash = self._calculate_temp_file_hash(temp_filename)
                    if expected_hash != actual_hash:
                        logger.error(f"文件校验失败: 期望 {expected_hash}, 实际 {actual_hash}")
                        if attempt < self.config.retry_times - 1:
                            logger.info("重试下载...")
                            continue
                        else:
                            os.remove(temp_filename)
                            return None
                
                # 重命名为最终文件名
                os.rename(temp_filename, expected_filename)
                # logger.info(f"文件已保存为: {expected_filename}")
                
                return os.path.abspath(expected_filename)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"下载失败 (尝试 {attempt + 1}): {e}")
                self._cleanup_temp_file(temp_filename)
                
                if attempt < self.config.retry_times - 1:
                    logger.info(f"等待 {self.config.retry_delay} 秒后重试...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error("所有重试都失败了")
                    return None
                    
            except Exception as e:
                logger.error(f"下载时发生错误 (尝试 {attempt + 1}): {e}")
                self._cleanup_temp_file(temp_filename)
                
                if attempt < self.config.retry_times - 1:
                    logger.info(f"等待 {self.config.retry_delay} 秒后重试...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error("所有重试都失败了")
                    return None
        
        return None
    
    def _calculate_temp_file_hash(self, temp_filename: str) -> Optional[str]:
        """计算临时文件的哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(temp_filename, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算临时文件哈希失败: {e}")
            return None
    
    def _cleanup_temp_file(self, temp_filename: str):
        """清理临时文件"""
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")

class SystemInfoManager:
    """系统信息管理类"""
    
    @staticmethod
    def get_system_architecture() -> str:
        """获取当前系统的架构"""
        arch = platform.machine().lower()
        arch_mapping = {
            'x86_64': 'x86_64',
            'amd64': 'x86_64',
            'aarch64': 'aarch64',
            'arm64': 'aarch64',
        }
        return arch_mapping.get(arch, arch)
    
    @staticmethod
    def get_python_version_tag() -> str:
        """获取与 .so 文件名兼容的 Python 版本标签"""
        major, minor = sys.version_info.major, sys.version_info.minor
        return f"{major}{minor}"
    
    @staticmethod
    def get_system_info() -> SystemInfo:
        """获取完整的系统信息"""
        return SystemInfo(
            architecture=SystemInfoManager.get_system_architecture(),
            python_version_tag=SystemInfoManager.get_python_version_tag(),
            platform_info=platform.platform(),
            python_version=sys.version
        )

class SOModuleLoader:
    """SO模块加载器"""
    
    def __init__(self, file_manager: FileManager, dependency_manager: Optional[DependencyManager] = None):
        self.file_manager = file_manager
        self.dependency_manager = dependency_manager or DependencyManager()
        self.max_dependency_retries = 3  # 最大依赖安装重试次数
    
    def find_so_file(self, base_name: str, py_ver_tag: str, 
                     arch: str, auto_download: bool = True,
                     network_manager: Optional[NetworkManager] = None) -> Optional[str]:
        """查找SO文件，如果不存在则尝试下载"""
        expected_filename = f"{base_name}.cpython-{py_ver_tag}-{arch}-linux-gnu.so"
        full_path = Path(expected_filename).resolve()
        
        if full_path.is_file():
            # logger.info(f"找到匹配的 SO 文件: {expected_filename}")
            
            if auto_download and network_manager:
                return self._handle_update_check(base_name, py_ver_tag, arch, full_path, network_manager)
            
            return str(full_path)
        else:
            logger.warning(f"未找到预期的 SO 文件: {expected_filename}")
            
            if auto_download and network_manager:
                return self._handle_download(base_name, py_ver_tag, arch, network_manager)
            
            self._list_so_files()
            return None
    
    def _handle_update_check(self, base_name: str, py_ver_tag: str, arch: str,
                           full_path: Path, network_manager: NetworkManager) -> str:
        """处理更新检查"""
        logger.info("检查是否需要更新...")
        current_version_info = self.file_manager.load_version_info()
        current_version = current_version_info.get('version') if current_version_info else None
        
        update_info = network_manager.check_server_update(base_name, py_ver_tag, arch, current_version)
        
        if update_info and update_info.get('has_update'):
            logger.info(f"发现新版本: {update_info.get('latest_version')}")
            logger.info(f"更新说明: {update_info.get('update_description', '无')}")
            
            return self._perform_update(base_name, py_ver_tag, arch, full_path, network_manager)
        
        return str(full_path)
    
    def _handle_download(self, base_name: str, py_ver_tag: str, arch: str,
                        network_manager: NetworkManager) -> Optional[str]:
        """处理下载"""
        logger.info("尝试从服务器下载SO文件...")
        
        current_version_info = self.file_manager.load_version_info()
        current_version = current_version_info.get('version') if current_version_info else None
        
        update_info = network_manager.check_server_update(base_name, py_ver_tag, arch, current_version)
        download_url, version_info = network_manager.request_so_download(base_name, py_ver_tag, arch)
        
        if download_url:
            downloaded_path = network_manager.download_so_file(base_name, py_ver_tag, arch, download_url)
            
            if downloaded_path and Path(downloaded_path).is_file():
                # logger.info(f"成功下载并保存SO文件: {downloaded_path}")
                
                if version_info:
                    self.file_manager.save_version_info(version_info)
                
                return downloaded_path
            else:
                logger.error("下载失败")
        else:
            logger.error("无法获取下载链接")
        
        return None
    
    def _perform_update(self, base_name: str, py_ver_tag: str, arch: str,
                       full_path: Path, network_manager: NetworkManager) -> str:
        """执行更新"""
        logger.info("开始下载更新...")
        download_url, version_info = network_manager.request_so_download(base_name, py_ver_tag, arch)
        
        if download_url:
            # 备份旧文件
            backup_filename = None
            if hasattr(self, 'update_config') and self.update_config.backup_old_files:
                backup_filename = self.file_manager.backup_file(full_path)
            
            # 下载新文件
            downloaded_path = network_manager.download_so_file(base_name, py_ver_tag, arch, download_url)
            
            if downloaded_path and Path(downloaded_path).is_file():
                logger.info(f"成功更新SO文件: {downloaded_path}")
                
                if version_info:
                    self.file_manager.save_version_info(version_info)
                
                # 删除备份文件
                if backup_filename and hasattr(self, 'update_config') and self.update_config.delete_backup_after_success:
                    if backup_filename.exists():
                        backup_filename.unlink()
                        logger.info("已删除备份文件")
                
                return downloaded_path
            else:
                logger.error("更新失败，恢复旧文件")
                if backup_filename:
                    self.file_manager.restore_file(backup_filename, full_path)
                return str(full_path)
        else:
            logger.error("无法获取更新下载链接")
            return str(full_path)
    
    def _list_so_files(self):
        """列出当前目录下的SO文件"""
        logger.info("当前目录下的 .so 文件:")
        try:
            for f in Path('.').glob('*.so'):
                logger.info(f"  - {f}")
        except Exception as e:
            logger.error(f"无法列出 .so 文件: {e}")
    
    def load_module(self, so_path: str, module_name: str) -> Optional[Any]:
        """加载SO模块，自动处理缺失依赖"""
        # logger.info(f"尝试使用模块名 '{module_name}' 加载")
        
        # 检查是否启用自动依赖安装
        auto_install = getattr(self, 'update_config', None) and getattr(self.update_config, 'auto_install_dependencies', True)
        
        for attempt in range(self.max_dependency_retries + 1):
            try:
                spec = importlib.util.spec_from_file_location(module_name, so_path)
                if spec is None:
                    logger.error("无法创建模块规范")
                    return None
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                return module
                
            except ImportError as e:
                error_msg = str(e)
                logger.error(f"ImportError: {error_msg}")
                
                # 尝试自动安装缺失的依赖
                if auto_install and attempt < self.max_dependency_retries:
                    logger.info(f"尝试自动安装缺失依赖 (第 {attempt + 1} 次)")
                    
                    if self.dependency_manager.auto_install_dependency(error_msg):
                        logger.info("依赖安装成功，重新尝试加载模块...")
                        continue
                    else:
                        logger.warning("依赖安装失败，继续重试...")
                        continue
                else:
                    if not auto_install:
                        logger.info("自动依赖安装已禁用")
                        module_name = self.dependency_manager.extract_missing_dependency(error_msg)
                        if module_name:
                            help_msg = self.dependency_manager.get_installation_help(module_name)
                            logger.info(f"建议: {help_msg}")
                    else:
                        logger.error("所有依赖安装尝试都失败了")
                        module_name = self.dependency_manager.extract_missing_dependency(error_msg)
                        if module_name:
                            help_msg = self.dependency_manager.get_installation_help(module_name)
                            logger.error(f"建议: {help_msg}")
                            
                            # 提供替代包建议
                            suggestions = self.dependency_manager.suggest_alternative_packages(module_name)
                            if suggestions:
                                logger.info(f"可能的替代包: {', '.join(suggestions)}")
                    return None
                    
            except Exception as e:
                logger.error(f"加载时发生错误: {e}")
                return None
        
        return None
    
    def call_function(self, module: Any, function_name: str = "main", 
                     args_list: Optional[list] = None) -> Optional[Any]:
        """调用模块中的函数"""
        try:
            if hasattr(module, function_name):
                target_func = getattr(module, function_name)
                
                if args_list is None:
                    if asyncio.iscoroutinefunction(target_func):
                        result = asyncio.run(target_func())
                    else:
                        result = target_func()
                else:
                    if asyncio.iscoroutinefunction(target_func):
                        result = asyncio.run(target_func(*args_list))
                    else:
                        result = target_func(*args_list)
                
                return result
            else:
                logger.error(f"未找到函数 '{function_name}'")
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                for attr in sorted(attrs):
                    logger.info(f"  - {attr}")
                return None

        except Exception as e:
            logger.error(f"调用函数时发生错误: {e}")
            return None

class LeaderKS:
    """主控制器类"""
    
    def __init__(self, config: ServerConfig, update_config: UpdateConfig):
        self.config = config
        self.update_config = update_config
        self.file_manager = FileManager()
        self.network_manager = NetworkManager(config)
        self.system_info = SystemInfoManager.get_system_info()
        self.dependency_manager = DependencyManager()
        self.so_loader = SOModuleLoader(self.file_manager, self.dependency_manager)
        self.so_loader.update_config = update_config
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置的有效性"""
        try:
            # 验证服务器配置
            if not self.config.base_url.startswith(('http://', 'https://')):
                logger.warning("服务器地址格式可能不正确")
            
            if self.config.timeout <= 0:
                logger.warning("超时时间应该大于0")
                self.config.timeout = 30
            
            if self.config.retry_times <= 0:
                logger.warning("重试次数应该大于0")
                self.config.retry_times = 3
            
            # 验证更新配置
            if self.update_config.auto_update and not self.update_config.backup_old_files:
                logger.warning("自动更新时建议启用文件备份")
                
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
    
    def diagnose_environment(self):
        """诊断运行环境"""
        logger.info("--- 环境诊断 ---")
        logger.info(f"Python 版本: {sys.version}")
        logger.info(f"平台详细信息: {self.system_info.platform_info}")
        logger.info(f"系统架构: {self.system_info.architecture}")
        logger.info(f"Python 版本标签: {self.system_info.python_version_tag}")
        
        # 检查关键依赖
        self._check_dependencies()
        
        # 检查并安装常见依赖
        logger.info("--- 依赖检查 ---")
        self.dependency_manager.check_and_install_common_dependencies()
    
    def _check_dependencies(self):
        """检查关键依赖"""
        try:
            import requests
            logger.info(f"✓ requests 版本: {requests.__version__}")
        except ImportError:
            logger.error("✗ requests 依赖未安装")
        
        try:
            import asyncio
            logger.info("✓ asyncio 可用")
        except ImportError:
            logger.error("✗ asyncio 依赖不可用")
    
    def run(self, so_base_name: str = "Kuaishou", custom_args: Optional[list] = None) -> int:
        """运行主程序"""
        start_time = time.time()
        
        try:
            logger.info(f"开始运行 LeaderKS")
            
            # 1. 环境诊断
            self.diagnose_environment()
            
            # 2. 查找SO文件
            # logger.info("开始查找SO文件...")
            so_file_path = self.so_loader.find_so_file(
                so_base_name, 
                self.system_info.python_version_tag, 
                self.system_info.architecture, 
                auto_download=self.update_config.auto_update,
                network_manager=self.network_manager
            )
            
            if not so_file_path:
                logger.error("致命错误: 找不到 .so 文件")
                return 1
            
            # 3. 尝试加载模块
            logger.info("开始加载SO模块...")
            module = self._load_module_with_fallback(so_file_path, so_base_name)
            if module is None:
                logger.error("所有加载方法都失败了")
                return 2
            
            # 4. 调用函数
            # logger.info("开始执行模块函数...")
            exit_code = self.so_loader.call_function(module, "main", custom_args)
            
            elapsed_time = time.time() - start_time
            logger.info(f"程序执行完成，耗时: {elapsed_time:.2f} 秒")
            
            if exit_code is not None:
                logger.info(f"脚本退出码: {exit_code}")
                return exit_code
            else:
                logger.info("脚本退出码: 2")
                return 2
                
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
            return 130
        except Exception as e:
            logger.error(f"程序运行出错: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return 1
    
    def _load_module_with_fallback(self, so_file_path: str, so_base_name: str) -> Optional[Any]:
        """使用多种方法尝试加载模块"""
        # 方法一：使用基础名称
        module = self.so_loader.load_module(so_file_path, so_base_name)
        
        # 方法二：尝试其他可能的模块名
        if module is None:
            possible_names = [
                "Kuaishou",  # 首字母大写
                "kuaishou",  # 全小写
                so_base_name.lower(),
            ]
            
            for name in possible_names:
                if module is None and name != so_base_name:
                    logger.info(f"尝试使用模块名: {name}")
                    module = self.so_loader.load_module(so_file_path, name)
                    if module:
                        logger.info(f"成功使用模块名 '{name}' 加载模块")
                        break
        
        return module

def create_default_config() -> Tuple[ServerConfig, UpdateConfig]:
    """创建默认配置"""
    server_config = ServerConfig()
    update_config = UpdateConfig()
    
    # 只检查服务器URL环境变量，其他功能直接开启
    if os.getenv('LEADERKS_SERVER_URL'):
        server_config.base_url = os.getenv('LEADERKS_SERVER_URL')
    
    # 直接开启自动更新和自动依赖安装
    update_config.auto_update = True
    update_config.auto_install_dependencies = True
    
    return server_config, update_config

def main():
    """主函数"""
    try:
        # 创建配置
        server_config, update_config = create_default_config()
        
        # 创建主控制器
        leader_ks = LeaderKS(server_config, update_config)
        
        # 运行程序
        exit_code = leader_ks.run("Kuaishou")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
