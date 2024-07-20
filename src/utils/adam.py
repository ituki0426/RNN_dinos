import numpy as np

def adam(param, param_adam, grad, lr , beta1, beta2, epsilon, L2_reg) -> None:
    """
    Adam optimizer
    Arguments:
    param -- parameters to be updated
        param["Wax"] = ...
        param["Waa"] = ...
        param["Wya"] = ...
        param["ba"] = ...
        param["by"] = ...
    param_adam -- Adam parameters
        param_adam["mWax"] = np.zeros_like(param["Wax"])
        param_adam["vWax"] = np.zeros_like(param["Wax"])
        param_adam["mWaa"] = np.zeros_like(param["Waa"])
        param_adam["vWaa"] = np.zeros_like(param["Waa"])
        param_adam["mWya"] = np.zeros_like(param["Wya"])
        param_adam["vWya"] = np.zeros_like(param["Wya"])
        param_adam["mba"] = np.zeros_like(param["ba"])
        param_adam["vba"] = np.zeros_like(param["ba"])
        param_adam["mby"] = np.zeros_like(param["by"])
        param_adam["vby"] = np.zeros_like(param["by"])
    grad -- gradients of parameters
        grad["dWax"] = np.zeros_like(param["Wax"])
        grad["dWaa"] = np.zeros_like(param["Waa"])
        grad["dWya"] = np.zeros_like(param["Wya"])
        grad["dba"] = np.zeros_like(param["ba"])
        grad["dby"] = np.zeros_like(param["by"])
    t -- current timestep
    """
    # Update m and v for Wax
    param_adam["mWax"] = beta1 * param_adam["mWax"] + (1 - beta1) * grad["dWax"]
    param_adam["vWax"] = beta2 * param_adam["vWax"] + (1 - beta2) * np.square(grad["dWax"])
    m_hat = param_adam["mWax"] / (1 - beta1)
    v_hat = param_adam["vWax"] / (1 - beta2)
    param["Wax"] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * param["Wax"])

    # Update m and v for Waa
    param_adam["mWaa"] = beta1 * param_adam["mWaa"] + (1 - beta1) * grad["dWaa"]
    param_adam["vWaa"] = beta2 * param_adam["vWaa"] + (1 - beta2) * np.square(grad["dWaa"])
    m_hat = param_adam["mWaa"] / (1 - beta1)
    v_hat = param_adam["vWaa"] / (1 - beta2)
    param["Waa"] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * param["Waa"])

    # Update m and v for Wya
    param_adam["mWya"] = beta1 * param_adam["mWya"] + (1 - beta1) * grad["dWya"]
    param_adam["vWya"] = beta2 * param_adam["vWya"] + (1 - beta2) * np.square(grad["dWya"])
    m_hat = param_adam["mWya"] / (1 - beta1)
    v_hat = param_adam["vWya"] / (1 - beta2)
    param["Wya"] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * param["Wya"])

    # Update m and v for ba
    param_adam["mba"] = beta1 * param_adam["mba"] + (1 - beta1) * grad["dba"]
    param_adam["vba"] = beta2 * param_adam["vba"] + (1 - beta2) * np.square(grad["dba"])
    m_hat = param_adam["mba"] / (1 - beta1)
    v_hat = param_adam["vba"] / (1 - beta2)
    param["ba"] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * param["ba"])

    # Update m and v for by
    param_adam["mby"] = beta1 * param_adam["mby"] + (1 - beta1) * grad["dby"]
    param_adam["vby"] = beta2 * param_adam["vby"] + (1 - beta2) * np.square(grad["dby"])
    m_hat = param_adam["mby"] / (1 - beta1)
    v_hat = param_adam["vby"] / (1 - beta2)
    param["by"] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * param["by"])
