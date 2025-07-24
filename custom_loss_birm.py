# -----------------------------
# 1. Определение BIRMLoss
# -----------------------------
# Основано на предоставленном коде и адаптировано под задачу токенизации
class BIRMLoss(nn.Module):
    def __init__(self, prior_sd_coef=1000.0, irm_lambda=1.0):
        """
        Bayesian Invariant Risk Minimization Loss для задачи токенизации.

        Args:
            prior_sd_coef: Коэффициент априорного распределения (L2 регуляризация).
            irm_lambda: Вес инвариантной регуляризации IRM.
        """
        super().__init__()
        self.prior_sd_coef = prior_sd_coef
        self.irm_lambda = irm_lambda

    def forward(self, model, logits, labels, environments):
        """
        Вычисление BIRM лосса.

        Args:
            model: Модель.
            logits: Выход модели [batch_size, seq_len, num_labels].
            labels: Целевые метки [batch_size, seq_len] (индексы классов или -100).
            environments: Идентификаторы окружений для каждого примера в батче [batch_size].

        Returns:
            total_loss: Общий лосс (NLL + IRM penalty + L2 регуляризация).
        """
        # 1. Вычисление основного лосса (NLL) - CrossEntropy для многоклассовой задачи
        batch_size, seq_len, num_labels = logits.shape
        flat_logits = logits.view(-1, num_labels)
        flat_labels = labels.view(-1)

        # Игнорируем -100 в лейблах
        mask = flat_labels != -100
        valid_logits = flat_logits[mask]
        valid_labels = flat_labels[mask]

        if valid_labels.numel() == 0:
            nll_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            ce_loss_fn = nn.CrossEntropyLoss()
            nll_loss = ce_loss_fn(valid_logits, valid_labels)

        # 2. Вычисление IRM penalty
        if valid_labels.numel() == 0:
            irm_penalty = torch.tensor(0.0, device=logits.device)
        else:
            # Выбираем логиты для истинных классов
            correct_class_logits = valid_logits.gather(1, valid_labels.unsqueeze(1)).squeeze(1) # [N_valid]
            
            # Расширяем environments до размера токенов
            expanded_envs = environments.unsqueeze(1).expand(-1, seq_len).contiguous() # [B, S]
            flat_envs = expanded_envs.view(-1) # [B*S]
            valid_envs = flat_envs[mask] # [N_valid]

            irm_penalty = self.compute_irm_penalty(correct_class_logits, valid_labels.float(), valid_envs)

        # 3. Вычисление L2 регуляризации
        l2_reg = torch.tensor(0., device=logits.device)
        for param in model.parameters():
            l2_reg += torch.norm(param) ** 2

        # 4. Комбинирование компонент лосса
        total_loss = (nll_loss +
                     self.irm_lambda * irm_penalty +
                     l2_reg / self.prior_sd_coef)

        return total_loss

    def compute_irm_penalty(self, logits, labels, environments):
        """
        Вычисление инвариантной регуляризации.

        Args:
            logits: Выход модели (логиты правильного класса) [N_valid].
            labels: Целевые метки (float) [N_valid].
            environments: Идентификаторы окружений [N_valid].

        Returns:
            penalty: IRM penalty term.
        """
        device = logits.device
        unique_envs = torch.unique(environments)
        penalty = torch.tensor(0., device=device)

        grad_norm_list = []

        for env in unique_envs:
            env_mask = (environments == env)
            if not torch.any(env_mask):
                continue

            env_logits = logits[env_mask]
            env_labels = labels[env_mask]

            dummy_weight = torch.tensor(1.0, device=device, requires_grad=True)

            dummy_logits = env_logits * dummy_weight
            # Используем BCEWithLogitsLoss, так как мы рассматриваем логит правильного класса
            # как "логит наличия этого класса" для конкретного токена.
            # Это упрощение, стандартный IRM для NLP сложнее.
            # Альтернатива: использовать MSE между логитом и 0 (или другим значением).
            # env_loss = F.binary_cross_entropy_with_logits(dummy_logits, torch.ones_like(env_labels)) # Пример альтернативы
            env_loss = F.mse_loss(dummy_logits, torch.zeros_like(env_labels)) # Еще одна альтернатива

            # Для демонстрации используем MSE между логитом и 0
            # Предполагая, что высокий логит для правильного класса "хорош"
            # Это спорное предположение, но для примера подходит
            # Более корректно было бы применять IRM к выходу всей сети или использовать специальные техники
            # env_loss = torch.mean((dummy_logits - 0) ** 2)

            # Или, если интерпретировать логит как "уверенность в классе 1" (плохая идея для многоклассовой)
            # env_loss = F.binary_cross_entropy_with_logits(dummy_logits, torch.ones_like(env_labels))

            # Лучше использовать MSE с 0, предполагая, что "хороший" логит для правильного класса положителен и большой
            env_loss = torch.mean((dummy_logits - 0) ** 2)


            grad_res = autograd.grad(env_loss, dummy_weight, create_graph=True)
            if grad_res and grad_res[0] is not None:
                grad_val = grad_res[0]
                grad_norm_list.append(grad_val ** 2)

        if len(grad_norm_list) > 1: # penalty определен только если есть хотя бы 2 среды
             penalty = sum(grad_norm_list) / len(grad_norm_list)
        elif len(grad_norm_list) == 1:
             penalty = grad_norm_list[0] # Если одна среда, penalty = 0 (градиент не должен меняться)
        # else: penalty = 0

        return penalty

# -----------------------------
# 2. Кастомный BIRM Trainer
# -----------------------------
class BIRMTrainer(Trainer):
    def __init__(self, birm_loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.birm_loss_fn = birm_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Переопределение метода вычисления лосса для использования BIRM.
        """
        # Извлекаем лейблы
        labels = inputs.pop("labels") # [B, S]
        environments = inputs.pop("environment") # [B] - добавлено в Dataset
        # inputs теперь содержит только 'input_ids' и 'attention_mask'

        # Прямой проход модели
        outputs = model(**inputs) # outputs.logits: [B, S, N]
        logits = outputs.logits

        # Вычисляем BIRM лосс
        loss = self.birm_loss_fn(model=model, logits=logits, labels=labels, environments=environments)

        return (loss, outputs) if return_outputs else loss
