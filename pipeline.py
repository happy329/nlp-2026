from classification.classifier_api import predict_class

text = "某科技公司发布人工智能芯片，主要用于大模型训练和推理。"

result = predict_class(text, model_name="textcnn")

print(result)