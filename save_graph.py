from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

class GenGraph:
    def __init__(self, title):
        self.title = title
        plt.figure(figsize=(20, 4))
        plt.title(title)

    def _save(self, save_image_path):        
        plt.savefig(str(save_image_path / '{}.png'.format(self.title)))

    def draw(self, save_path, data):
        plt.plot(data)
        self._save(save_path)

    def __del__(self):
        plt.clf()

class BasicGenGraph(GenGraph):
    def draw(self, save_path, data):
        plt.plot(data['real_value'], label='real')

        scatter_x = []
        scatter_y = []
        for i, (real_value, anomaly_label) in enumerate(
                zip(data['real_value'], data['anomaly_label'])):
            if anomaly_label == 1:
                scatter_x.append(i)
                scatter_y.append(real_value)
        plt.scatter(scatter_x, scatter_y, color='0.5', edgecolor="r", label="anomaly point")
        plt.legend(loc='upper right')
        self._save(save_path)
