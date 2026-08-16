import torch


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)

        matches = torch.eq(predicted, labels)
        correct += torch.sum(matches).item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            total += labels.size(0)

            matches = torch.eq(predicted, labels)
            correct += torch.sum(matches).item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def train_one_epoch_ewc(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    ewc_tasks,
    ewc_lambda=1000.0
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = criterion(
            outputs,
            labels
        )

        ewc_penalty = torch.tensor(
            0.0,
            device=device
        )

        for ewc in ewc_tasks:
            ewc_penalty += ewc.penalty(model)

        loss = (
            classification_loss
            + ewc_lambda * ewc_penalty
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            classification_loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        matches = torch.eq(predicted, labels)
        correct += torch.sum(matches).item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def train_one_epoch_lwf(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    lwf=None
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = criterion(
            outputs,
            labels
        )

        if lwf is not None:
            distillation_loss = lwf.distillation_loss(
                model,
                images
            )

            loss = (
                classification_loss
                + distillation_loss
            )

        else:
            loss = classification_loss

        loss.backward()
        optimizer.step()

        total_loss += (
            classification_loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        matches = torch.eq(predicted, labels)
        correct += torch.sum(matches).item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def train_one_epoch_replay_lwf(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    lwf=None,
    distillation_weight=1.0
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = criterion(
            outputs,
            labels
        )

        if lwf is not None:
            distillation_loss = lwf.distillation_loss(
                model,
                images
            )

            loss = (
                classification_loss
                + distillation_weight
                * distillation_loss
            )

        else:
            loss = classification_loss

        loss.backward()
        optimizer.step()

        total_loss += (
            classification_loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        matches = torch.eq(predicted, labels)
        correct += torch.sum(matches).item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy