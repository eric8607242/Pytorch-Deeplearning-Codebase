import os
import time

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from utils import get_logger, get_writer, set_random_seed, get_optimizer, get_lr_scheduler, resume_checkpoint, AverageMeter, save, LoggingTracker
from criterion import get_criterion
from dataflow import get_dataloader
from model import get_model_class

class TrainingAgent:
    def __init__(self, config, title):
        self.config = config

        self.logger = get_logger(config["logs_path"]["logger_path"])
        self.writer = get_writer(
            title,
            config["train"]["random_seed"],
            config["logs_path"]["writer_path"])

        if self.config["train"]["random_seed"] is not None:
            set_random_seed(config["train"]["random_seed"])

        self.device = torch.device(config["train"]["device"])

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.config["dataset"]["dataset"],
                self.config["dataset"])


        model_class = get_model_class(self.config["agent"]["model_agent"])
        model = model_class(self.config["model"])
        self.model = model.to(self.device)
        self.model = self._parallel_process(self.model)

        criterion = get_criterion(config["agent"]["criterion_agent"], config["criterion"])
        self.criterion = criterion.to(self.device)
        self.criterion = self._parallel_process(self.criterion)

        self._optimizer_init(self.model)
        self.epochs = config["train"]["epochs"]
        self.start_epochs = 0

        self.logging_tracker = LoggingTracker(self.logger, self.writer, self.config, title)

        
    def fit(self):
        """
        """
        start_time = time.time()
        self.logger.info("Training process start!")

        self.train_loop()

    def train_loop(self):
        best_val_performance = -float("inf")

        for epoch in range(self.start_epochs, self.epochs):
            self.logger.info(f"Start to train for epoch {epoch}")
            self.logger.info(f"Learning Rate : {self.optimizer.param_groups[0]['lr']:.8f}")

            self._training_step(self.model, self.train_loader, epoch)
            val_loss = self._validate_step(self.model, self.val_loader, epoch)

            self.evaluate(self.model, self.test_loader, epoch)

            save(
                self.model,
                os.path.join(
                    self.config["experiment_path"]["checkpoint_root_path"],
                    f"checkpoints_{epoch}.pth"),
                None,
                None,
                None,
                None)


    def _training_step(self, model, train_loader, epoch, print_freq=20):
        losses = AverageMeter()

        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            N = X.shape[0]
            self.optimizer.zero_grad()
            out = model(X)
            
            loss = self.criterion(out, y)
            loss.backward()

            self.optimizer.step()

            losses.update(loss.item(), N)
            if (step > 1 and step % print_freq == 0) or (step == len(train_loader) - 1):
                self.logger.info(f"Train : [{(epoch+1):3d}/{self.epochs}] "
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f}")

        self.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)
        self.logger.info(
            f"Train: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f}" 
            f"Time {time.time() - start_time:.2f}")

    def _validate_step(self, model, val_loader, epoch, print_freq=20):
        losses = AverageMeter()

        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                N = X.shape[0]
                out = model(X)
                
                loss = self.criterion(out, y)
                losses.update(loss.item(), N)

                if (step > 1 and step % print_freq == 0) or (step == len(val_loader) - 1):
                    self.logger.info(f"Val: [{(epoch+1):3d}/{self.epochs}] "
                                     f"Step {step:3d}/{len(val_loader)-1:3d} Loss {losses.get_avg():.3f}")

            self.writer.add_scalar("Val/_loss/", losses.get_avg(), epoch)
            self.logger.info(
                f"Val: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f}" 
                f"Time {time.time() - start_time:.2f}")

        return losses.get_avg()

    def evaluate(self, model, test_loader, epoch):
        self.logger.info("Evaluating starting ------------------")
        pass

    def _optimizer_init(self, model, criterion=None):
        parameters_set = [{"params": model.parameters()}]
        if criterion is not None:
            parameters_set.append({"params": criterion.parameters()})

        self.optimizer = get_optimizer(
            parameters_set,
            self.config["optim"]["optimizer"],
            learning_rate=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["weight_decay"],
            logger=self.logger,
            momentum=self.config["optim"]["momentum"],
            alpha=self.config["optim"]["alpha"],
            beta=self.config["optim"]["beta"])

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.config["optim"]["scheduler"],
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.config["optim"]["decay_step"],
            decay_ratio=self.config["optim"]["decay_ratio"],
            total_epochs=self.config["train"]["epochs"])

    def _resume(self, model, criterion):
        """ Load the checkpoint of model, optimizer, and lr scheduler.
        """
        if self.config["train"]["resume"]:
            self.start_epochs = resume_checkpoint(
                    model,
                    self.config["experiment_path"]["resume_path"],
                    criterion,
                    None,
                    None)
            self.logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config["experiment_path"]["resume_path"], self.start_epochs))

    def _parallel_process(self, model):
        if self.device.type == "cuda" and self.config["train"]["ngpu"] >= 1:
            return nn.DataParallel(
                model, list(range(self.config["train"]["ngpu"])))
        else:
            return model
