from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ELyolo.yaml")
    data = "VisDrone.yaml"  #VisDrone.yaml   DOTAv1.5.yaml （8，4）
    name = 'exp-测试-'
    model.info(verbose=True)  # 查看通道数变化


    model.train(data=data,
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                epochs=300,
                batch=16,
                imgsz=640,
                device='0',
                workers=16,
                name=name,
                exist_ok=False,  # 如果为 True，则允许覆盖现有 project/name 目录。
                optimizer='SGD',  # SGD、Adam、AdamW、NAdam、RAdam、RMSProp 等，或 auto
                close_mosaic=10,  # 在最后 N 个 epoch 中禁用马赛克数据增强，以在完成之前稳定训练。设置为 0 将禁用此功能。
                project='runs/train/autodl',
                plots=True,
                # resume=True,
                seed=2022116
                )
"""
    python train.py && /usr/bin/shutdown    # 用&&拼接表示前边的命令执行成功后才会执行shutdown。请根据自己的需要选择
"""