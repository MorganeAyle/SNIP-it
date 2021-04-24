import foolbox as fb
import torch

def get_attack(method_name):

    att = fb.attacks

    print("> PERFORMING ADV ATTACK", method_name)

    switcher = {
        "CarliniWagner": att.L2CarliniWagnerAttack,
        "LinfPGD": att.LinfPGD,
        "L1FastGradientAttack": att.L1FastGradientAttack,
        "L2DeepFoolAttack": att.L2DeepFoolAttack,
        "FGSM": att.FGSM,
        "DDNAttack": att.DDNAttack,
        "SaltAndPepperNoiseAttack": att.SaltAndPepperNoiseAttack,
        "L2RepeatedAdditiveGaussianNoiseAttack": att.L2RepeatedAdditiveGaussianNoiseAttack,
    }
    attack = switcher.get(method_name, f"{method_name} not recognised")()
    return attack


def construct_adversarial_examples(im, crit, method, model, device, exclude_wrong_predictions, targeted, epsilons):
    bounds = (im.min().item(), im.max().item())
    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device)

    im = im.to(device)
    crit = crit.to(device)
    probs = model.forward(im)
    predictions = probs.argmax(dim=-1)

    if exclude_wrong_predictions:
        selection = predictions == crit
        im = im[selection]
        crit = crit[selection]
        predictions = predictions[selection]

    if targeted:
        target = 1
        selection = crit != target
        im = im[selection]
        predictions = predictions[selection]
        miss_classifications = torch.tensor([target] * len(im))
        crit = fb.criteria.TargetedMisclassification(
            miss_classifications)

    attack = get_attack(method)

    return attack(fmodel, im, crit, epsilons=epsilons), predictions

    