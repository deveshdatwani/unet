from matplotlib import pyplot as plt


def display_inference(prediction, mask, instance):
    plt.subplot(1,3,1)
    plt.imshow(instance[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(prediction[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(mask[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
    plt.show()
    
    return None