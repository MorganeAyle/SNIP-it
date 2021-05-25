def lipschitz_opt_lb(model, initial_max=None, num_iter=100):
    """ Compute lower bound of the Lipschitz constant with optimization on gradient norm
    INPUTS:
        * `initial_max`: initial seed for the SGD
        * `num_iter`: number of SGD iterations
    If initial_max is not provided, the model must have an input_size attribute
    which consists in a list of torch.Size.
    """
    use_cuda = next(model.parameters()).is_cuda
    if initial_max is None:
        input_size = model.input_sizes[0]
        initial_max = torch.randn(input_size)
    mone = torch.Tensor([-1])
    if use_cuda:
        initial_max = initial_max.cuda()
        mone = mone.cuda()
    v = nn.Parameter(initial_max, requires_grad=True)

    optimizer = optim.Adam([v], lr=1e-3)
    # optimizer = optim.SGD([v], lr=1e-3, momentum=0.9,
    #         nesterov=False)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
            factor=0.5, patience=50, cooldown=10, threshold=1e-6, eps=1e-6,
            verbose=10)

    it = 0

    loss_mean = []
    max_loss = 0

    while it < num_iter:
        # v = initial_max
        optimizer.zero_grad()
        loss = gradient_norm(model, v)**2
        loss.backward(mone)
        optimizer.step()
        loss_mean.append(np.sqrt(loss.data[0]))
        if loss.data[0] > max_loss:
            max_loss = loss.data[0]
        if it%10 == 0:
            print('[{}] {:.4f} (max: {:.4f})'.format(it,
                np.mean(loss_mean), math.sqrt(max_loss)))
            schedule.step(np.mean(loss_mean))
            loss_mean = []

        del loss  # Free the graph
        # v.data.clamp_(-2, 2)

        it += 1

    return gradient_norm(model, v).data[0]