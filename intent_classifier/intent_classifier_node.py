#!/usr/bin/env python3
"""
FastText意图识别ROS 2节点 - 兼容新版API
订阅：/AAA (std_msgs/String) - 原始语音文本
发布：/BBB (std_msgs/String) - 识别出的意图标签
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import fasttext
import threading

class IntentClassifierNode(Node):
    def __init__(self):
        super().__init__('intent_classifier')
        
        # 参数声明
        self.declare_parameter('model_path', 'intent_v1.bin')
        self.declare_parameter('input_topic', '/asr_result')
        self.declare_parameter('output_topic', '/BBB')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('enable_preprocessing', True)
        
        # 获取参数
        model_path = self.get_parameter('model_path').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.enable_preprocessing = self.get_parameter('enable_preprocessing').value
        
        # 加载FastText模型
        self.get_logger().info(f'正在加载模型: {model_path}')
        try:
            self.model = fasttext.load_model(model_path)
            self.get_logger().info('✓ 模型加载成功')
            
            # 打印模型信息（调试用）
            self.print_model_info()
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')
            raise
        
        # 创建发布者和订阅者
        self.intent_publisher = self.create_publisher(
            String, 
            output_topic, 
            10
        )
        
        self.text_subscriber = self.create_subscription(
            String,
            input_topic,
            self.text_callback,
            10
        )
        
        # 统计信息
        self.stats = {
            'total_messages': 0,
            'successful_predictions': 0,
            'low_confidence': 0
        }
        
        # 定时器：定期打印统计信息
        self.create_timer(30.0, self.print_statistics)
        
        self.get_logger().info(f'节点已启动，订阅: {input_topic}，发布: {output_topic}')
        self.get_logger().info(f'置信度阈值: {self.confidence_threshold}')
    
    def print_model_info(self):
        """打印模型信息 - 兼容新版API"""
        try:
            self.get_logger().info('=' * 50)
            self.get_logger().info('模型信息:')
            
            # 方法1：尝试获取标签
            try:
                # 使用测试预测来获取标签信息
                test_result = self.model.predict("测试", k=5)
                self.get_logger().info(f'  模型响应正常，支持预测功能')
            except Exception as e:
                self.get_logger().warn(f'  测试预测时出错: {e}')
            
            # 方法2：打印模型文件信息
            import os
            if hasattr(self.model, '_model_file'):
                model_size = os.path.getsize(self.model._model_file) / 1024 / 1024
                self.get_logger().info(f'  模型文件大小: {model_size:.2f} MB')
            
            # 方法3：检查FastText版本
            import fasttext
            self.get_logger().info(f'  FastText版本: {fasttext.__version__}')
            
            self.get_logger().info('=' * 50)
        except Exception as e:
            self.get_logger().warn(f'获取模型信息失败: {e}')
            self.get_logger().info('=' * 50)
    
    def preprocess_text(self, text):
        """
        预处理文本，与训练时保持一致
        """
        if not self.enable_preprocessing:
            return text
        
        # 示例预处理步骤（根据你的训练数据处理方式调整）
        processed_text = text
        
        # 1. 转换为小写（如果是英文或中英混合）
        # processed_text = processed_text.lower()
        
        # 2. 移除多余空格
        import re
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # 3. 如果你的训练数据是按字空格的，可以启用以下处理
        # processed_text = ' '.join(list(processed_text))
        
        return processed_text
    
    def text_callback(self, msg):
        """
        处理收到的文本消息
        """
        self.stats['total_messages'] += 1
        
        raw_text = msg.data
        self.get_logger().debug(f'收到文本: "{raw_text}"')
        
        # 1. 预处理文本
        processed_text = self.preprocess_text(raw_text)
        
        # 2. 使用模型预测
        try:
            # 预测top-2个标签
            predictions = self.model.predict(processed_text, k=2)
            
            # 新版API返回的是元组 (labels, probabilities)
            labels = predictions[0]
            probabilities = predictions[1]
            
            # 提取第一个预测结果
            top_label = labels[0]
            top_confidence = probabilities[0]
            
            # 移除标签前缀（如果存在）
            if top_label.startswith('__label__'):
                top_label = top_label.replace('__label__', '')
            
            # 3. 检查置信度
            if top_confidence < self.confidence_threshold:
                self.stats['low_confidence'] += 1
                self.get_logger().warn(
                    f'低置信度预测: "{raw_text}" -> {top_label} ({top_confidence:.3f})'
                )
                
                # 发布特殊标签
                output_msg = String()
                output_msg.data = 'UNKNOWN'
                self.intent_publisher.publish(output_msg)
                return
            
            # 4. 发布识别结果
            output_msg = String()
            output_msg.data = top_label
            
            self.intent_publisher.publish(output_msg)
            self.stats['successful_predictions'] += 1
            
            # 记录日志
            self.get_logger().info(
                f'预测成功: "{raw_text}" -> {top_label} ({top_confidence:.3f})'
            )
            
            # 5. 记录备选标签（调试用）
            if len(labels) > 1:
                alt_label = labels[1]
                if alt_label.startswith('__label__'):
                    alt_label = alt_label.replace('__label__', '')
                alt_prob = probabilities[1]
                self.get_logger().debug(f'  备选标签: {alt_label} ({alt_prob:.3f})')
                
        except Exception as e:
            self.get_logger().error(f'预测失败: {e}, 文本: "{raw_text}"')
    
    def print_statistics(self):
        """打印统计信息"""
        total = self.stats['total_messages']
        if total == 0:
            return
            
        success_rate = (self.stats['successful_predictions'] / total) * 100
        low_conf_rate = (self.stats['low_confidence'] / total) * 100
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('节点统计:')
        self.get_logger().info(f'  总处理消息: {total}')
        self.get_logger().info(f'  成功预测: {self.stats["successful_predictions"]} ({success_rate:.1f}%)')
        self.get_logger().info(f'  低置信度: {self.stats["low_confidence"]} ({low_conf_rate:.1f}%)')
        self.get_logger().info('=' * 50)
    
    def destroy_node(self):
        """节点销毁前的清理"""
        self.print_statistics()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # 创建节点
    node = IntentClassifierNode()
    
    try:
        # 运行节点
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断')
    finally:
        # 清理
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()