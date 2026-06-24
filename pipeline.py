from classification.classifier_api import predict_class


def run_pipeline(text):
    class_result = predict_class(
        text,
        model_name="bilstm_attention",
        top_k=5
    )

    result = {
        "classification": class_result
    }

    return result