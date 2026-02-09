#!/usr/bin/env python3
"""
意图识别节点启动文件
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_name = 'intent_classifier'
    
    # 获取包目录
    package_dir = get_package_share_directory(package_name)
    
    # 模型文件路径（假设放在包的models目录下）
    model_path = os.path.join(
        package_dir, 
        'models', 
        'intent_v1.bin'
    )
    
    # 创建节点
    intent_classifier_node = Node(
        package=package_name,
        executable='intent_classifier_node',
        name='intent_classifier',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'input_topic': '/asr_result',           # 订阅的话题
            'output_topic': '/BBB',          # 发布的话题
            'confidence_threshold': 0.3,     # 置信度阈值
            'enable_preprocessing': True     # 启用预处理
        }]
    )
    
    return LaunchDescription([
        intent_classifier_node
    ])