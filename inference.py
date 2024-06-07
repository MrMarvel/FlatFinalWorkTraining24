import onnx
import onnxruntime
import torch
from PIL import Image
import torchvision.transforms as transforms

device = 'cuda' if onnxruntime.get_device().lower() == 'gpu' else 'cpu'


def main():
    model_name = './trained/best.onnx'
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    image = Image.open("./datasets/yolo_final/valid/images/62_jpg.rf.2edbf1d30736a36a2c8cc5f639ae05a4.jpg")
    resize = transforms.Compose(
        [transforms.Resize((640, 640)), transforms.ToTensor()])
    image = resize(image)
    image = image.unsqueeze(0)  # add fake batch dimension
    image = image.to(device)

    ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = onnxruntime.InferenceSession(model_name, providers=ep_list)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs
    max = float('-inf')
    max_index = -1
    for i in range(0, len(ort_outs[0][0])):
        if (ort_outs[0][0][i] > max):
            max = ort_outs[0][0][i]
            max_index = i
    print(max_index)


if __name__ == '__main__':
    main()
