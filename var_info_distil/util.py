import torch

from ignite.metrics import Average, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar

# Function for moving image and label from CPU to device used for training.
def prepare_batch(batch, device):
    image, label = batch
    image = image.to(device)
    label = label.to(device)
    return image, label

# Function to attach the progress bar, loss metric, and the accuracy metrics to ignite engines.
def attach_pbar_and_metrics(trainer, evaluator):
    loss_metric = Average(output_transform=lambda output: output["loss"])
    accuracy_metric = Accuracy(output_transform=lambda output: (output["logit"], output["label"]))
    pbar = ProgressBar()
    loss_metric.attach(trainer, "loss")
    accuracy_metric.attach(trainer, "accuracy")
    accuracy_metric.attach(evaluator, "accuracy")
    pbar.attach(trainer)

class LearningRateUpdater(object):
    """
    When called, this class updates the counter for the PyTorch learning rate scheduler.
    
    :param torch.optim.scheduler: PyTorch learning rate scheduler.
    """
    def __init__(self, lr_scheduler): 
        self.lr_scheduler = lr_scheduler
    
    def __call__(self, engine):
        self.lr_scheduler.step()

class MetricLogger(object):
    """
    When called, this class logs the metrics collected from the Ignite engines, i.e., trainer and evaluator.
    
    :param torch.utils.data.DataLoader: Data loader needed for evaluation on the validation dataset. 
    """
    def __init__(self, evaluator, eval_loader):
        self.evaluator = evaluator
        self.eval_loader = eval_loader
    
    def __call__(self, engine):
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        loss = engine.state.metrics["loss"]
        accuracy = engine.state.metrics["accuracy"]

        print(f"Epoch {epoch}/{max_epochs}")
        print(f"Train statistics: loss: {loss:.4f}, accuracy: {accuracy:3.4f}")

        self.evaluator.run(self.eval_loader)
        eval_accuracy = self.evaluator.state.metrics["accuracy"]
        print(f"Evaluation statistics: accuracy: {eval_accuracy:3.4f}")
    


class BatchEvaluator(object):
    """
    When called, this class evaluates a mini-batch of samples to report the performance of the model being trained.
    
    :param torch.nn.Module model: model for training.
    :param int device: device to proceed the training.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, evaluator, batch):
        self.model.eval()
        image, label = prepare_batch(batch, self.device)
        with torch.no_grad():
            logit = self.model(image)[0]
        return {"logit": logit, "label": label}


class BatchUpdaterWithoutTransfer(object):
    """
    When called, this class uses a mini-batch of samples to train the model without any transfer learning.
    
    :param torch.nn.Module model: model for training.
    :param torch.optim.Optimizer optimizer: optimizer to use for training the model.
    :param torch.nn.Module criterion: module to generate the loss function for the model. 
    :param int device: device to proceed the training.
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def __call__(self, trainer, batch):
        self.model.train()
        self.optimizer.zero_grad()
        image, label = prepare_batch(batch, self.device)
        logit = self.model(image)[0]
        loss = self.criterion(logit, label)
        loss.backward()
        self.optimizer.step()

        output = {"logit": logit, "label": label, "loss": loss.item()}
        return output


class BatchUpdaterWithTransfer(object):
    """
    When called, this class uses a mini-batch of samples to train the student model based on transfer learning from the 
    teacher model.
    
    :param torch.nn.Module model: student model for training.
    :param torch.nn.Module teacher_model: teacher model for training.
    :param torch.optim.Optimizer optimizer: optimizer to use for training the student model.
    :param torch.nn.Module criterion: module to generate the loss function for the student model. 
    :param int device: device to proceed the training.
    
    """

    def __init__(self, model, teacher_model, optimizer, criterion, device):
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def __call__(self, trainer, batch):
        """
        :param ignite.engine.Engine trainer: Ignite engine class for training the model.
        :param tuple(torch.Tensor) batch: tuple of samples for training the model.
        """

        self.model.train()
        self.optimizer.zero_grad()
        image, label = prepare_batch(batch, self.device)
        with torch.no_grad():
            teacher_logit, teacher_features = self.teacher_model(image)

        logit, teacher_feature_preds = self.model(image)
        loss = self.criterion(
            logit=logit,
            label=label,
            teacher_feature_preds=teacher_feature_preds,
            teacher_logit=teacher_logit,
            teacher_features=teacher_features,
        )
        loss.backward()
        self.optimizer.step()
        
        return {"logit": logit, "label": label, "loss": loss.item()}
