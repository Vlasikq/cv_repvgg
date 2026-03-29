# Фабрика моделей - создаём нужную модель по имени из конфига

from src.models.repvgg import build_repvgg_a0
from src.models.resnet import build_resnet20
from src.models.vgg import build_vgg16_cifar


def build_model(config: dict):
    """Создаём модель по конфигу. Возвращает готовую к обучению модель."""
    name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    use_identity = config["model"].get("use_identity", True)
    use_1x1 = config["model"].get("use_1x1", True)

    if name == "repvgg_a0":
        return build_repvgg_a0(
            num_classes=num_classes,
            use_identity=use_identity,
            use_1x1=use_1x1,
        )
    elif name == "resnet20":
        return build_resnet20(num_classes=num_classes)
    elif name == "vgg16_cifar":
        return build_vgg16_cifar(num_classes=num_classes)
    else:
        raise ValueError(f"Неизвестная модель: {name}")
