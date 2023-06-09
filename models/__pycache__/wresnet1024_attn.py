U
    ��`�  �                   @   s�   d dl Z d dlmZ d dlmZ d dlmZmZmZ G dd� dej	�Z
G dd� dej	�ZG dd	� d	ej	�ZG d
d� dej	�ZdS )�    N)�wrn38)�
ConvBnRelu�ConvTransposeBnRelu�initialize_weightsc                       s&   e Zd Zd� fdd�	Zdd� Z�  ZS )�ChannelAttention�   c              	      s\   t t| ���  t�d�| _t�t�||| �t�|| �t�	� t�|| |�t�
� �| _d S �N�   )�superr   �__init__�nn�AdaptiveAvgPool2d�avg_pool�
Sequential�Linear�BatchNorm1d�ReLU�Sigmoid�attn)�self�input_channels�	reduction��	__class__� �</home/hci/Felix2021/Project250321/models/wresnet1024_attn.pyr      s    �zChannelAttention.__init__c                 C   sF   |� � \}}}}| �|��||�}| �|��||dd�}||�|� S r   )�sizer   �viewr   �	expand_as)r   �x�b�c�_�yr   r   r   �forward   s    zChannelAttention.forward)r   ��__name__�
__module__�__qualname__r   r$   �__classcell__r   r   r   r   r      s   r   c                       s&   e Zd Zd� fdd�	Zdd� Z�  ZS )�SpatialAttentionr   c                    s�   t t| ���  || }t�tj||dd�t�|�t�� tj||dddd�t�|�t�� tj||dddd�t�|�t�� tj|ddd�t�� �| _	d S )Nr	   ��kernel_size�   )r,   �padding�dilation�   )
r
   r*   r   r   r   �Conv2d�BatchNorm2dr   r   �sattn)r   r   r   Zreduction_channelsr   r   r   r      s    �zSpatialAttention.__init__c                 C   s   | � |�}|| S �N)r3   )r   r   r#   r   r   r   r$   /   s    
zSpatialAttention.forward)r   r%   r   r   r   r   r*      s   r*   c                       s&   e Zd Zd� fdd�	Zdd� Z�  ZS )	�	Attentionr   TFc                    sB   t t| ���  d\| _| _|r,t||d�| _|r>t||d�| _d S )N)NN)r   )r
   r5   r   �cattnr3   r   r*   )r   r   r   r6   r3   r   r   r   r   5   s    zAttention.__init__c                 C   s,   | j d k	r| � |�}| jd k	r(| �|�}|S r4   )r3   r6   )r   r   r   r   r   r$   =   s
    



zAttention.forward)r   TFr%   r   r   r   r   r5   4   s   r5   c                       s,   e Zd Zdd� Z� fdd�Zdd� Z�  ZS )�LixNetc                 C   s   dS )Nz"Wider ResNet 1024 + CATTN (Linear)r   )r   r   r   r   �get_nameH   s    zLixNet.get_namec                    s�  t t| ���  ddddddg}|jj}|jjj}t|dd�| _tj	|d	 | |d
 d
dd�| _
tj	|d | |d d
dd�| _tj	|d | |d d
dd�| _t|d
 |d dd�| _t|d |d  |d dd�| _t|d |d  |d dd�| _t|d dddd�| _t|d dddd�| _t|d dddd�| _t�t|d |d d
d	d�t|d |d dd
d�tj	|d d||dk�r�d
nd	dd��| _t| j| j| j
� t| j| j| j� t| j| j| j� t| j� d S )Ni   i   i   i   �   �   T)Z
pretrainedr   r	   F)r,   �bias�   �   �   r+   r-   r   )r   r6   r3   )r,   r.   )r,   r.   r;   )r
   r7   r   �MODEL�ENCODED_FRAMES�EXTRA�FINAL_CONV_KERNELr   r   r1   �conv_x8�conv_x2�conv_x1r   �up8�up4�up2r5   �attn8�attn4�attn2r   r   �finalr   )r   �config�channels�framesZfinal_conv_kernelr   r   r   r   K   s6    
     ��	zLixNet.__init__c           	      C   s�   g g g   }}}|D ]2}| � |�\}}}|�|� |�|� |�|� q| �tj|dd��}| �tj|dd��}| �tj|dd��}| �|�}| �|�}| �	tj||gdd��}| �
|�}| �tj||gdd��}| �|�}| �|�S )Nr	   )�dim)r   �appendrC   �torch�catrD   rE   rF   rI   rG   rJ   rH   rK   rL   )	r   r   Zx1sZx2sZx8s�xi�x1�x2Zx8r   r   r   r$   l   s     





zLixNet.forward)r&   r'   r(   r8   r   r$   r)   r   r   r   r   r7   G   s   !r7   )rR   �torch.nnr   Znetworks.wider_resnetr   Znetworks.helperr   r   r   �Moduler   r*   r5   r7   r   r   r   r   �<module>   s   