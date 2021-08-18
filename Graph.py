from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import cv2

class GraphType(Enum):
    LINE = 0
    WIREFRAME = 1
    SURFACE = 2
    CONTOUR = 3

class Graph:
    # proj: either "2d" or "3d"
    def __init__(self, 
    dim3 = True, graph_type = GraphType.LINE,
    apply_equation = True,
    x_equation = np.linspace(-10, 10, 25),
    y_equation = np.linspace(-10, 10, 25),
    z_equation = lambda x, y: (np.cos(np.sqrt(x**2 + y**2)))):

        # graph details
        self.__projection = "3d" if dim3 else "2d"
        self.__figure = plt.figure()
        if dim3:
            self.__graph = self.__figure.add_subplot(111, projection = "3d")
        else:
            self.__graph = plt
        self.__graph_type = graph_type

        # axis details
        if apply_equation:
            if dim3:
                x, y = np.meshgrid(x_equation, y_equation)
                self.__x_equation = x
                self.__y_equation = y
                self.__z_equation = z_equation(x, y)
            else:
                self.__x_equation = x_equation
                self.__y_equation = y_equation(x_equation)
        else:
            if dim3:
                x, y, z = np.meshgrid(x_equation, y_equation, z_equation)
                self.__x_equation = x
                self.__y_equation = y
                self.__z_equation = z
            else:
                self.__x_equation = x_equation
                self.__y_equation = y_equation

    @property
    def x_equation(self):
        return self.__x_equation

    @x_equation.setter
    def x_equation(self, x_equation):
        self.__x_equation = x_equation

    @property
    def y_equation(self):
        return self.__y_equation

    @y_equation.setter
    def y_equation(self, y_equation):
        self.__y_equation = y_equation

    @property
    def z_equation(self):
        return self.__z_equation

    @z_equation.setter
    def z_equation(self, z_equation):
        self.__z_equation = z_equation

    @property
    def projection(self):
        return self.__projection

    @projection.setter
    def projection(self, projection):
        self.__projection = projection
        if projection == "3d":
            self.__graph = plt.axes(projection = projection)
        else:
            self.__graph = plt
    
    @property
    def graph_type(self):
        return self.__graph_type

    @graph_type.setter
    def graph_type(self, graph_type):
        self.__graph_type = graph_type

    def render(self):
        if self.__projection == "2d":
            self.__graph.plot(self.__x_equation, self.__y_equation)
        elif self.__projection == "3d":
            if self.__graph_type == GraphType.LINE:
                self.__graph.plot(self.__x_equation, self.__y_equation, self.__z_equation)
            elif self.__graph_type == GraphType.WIREFRAME:
                self.__graph.plot_wireframe(self.__x_equation, self.__y_equation, self.__z_equation)
            elif self.__graph_type == GraphType.SURFACE:
                self.__graph.plot_surface(self.__x_equation, self.__y_equation, self.__z_equation)
            elif self.__graph_type == GraphType.CONTOUR:
                self.__graph.contour(self.__x_equation, self.__y_equation, self.__z_equation)
        
        plt.show()

    def render_to_image(self):
        if self.__projection == "2d":
            self.__graph.plot(self.__x_equation, self.__y_equation)
        elif self.__projection == "3d":
            if self.__graph_type == GraphType.LINE:
                self.__graph.plot(self.__x_equation, self.__y_equation, self.__z_equation)
            elif self.__graph_type == GraphType.WIREFRAME:
                self.__graph.plot_wireframe(self.__x_equation, self.__y_equation, self.__z_equation)
            elif self.__graph_type == GraphType.SURFACE:
                self.__graph.plot_surface(self.__x_equation, self.__y_equation, self.__z_equation)
            elif self.__graph_type == GraphType.CONTOUR:
                self.__graph.contour(self.__x_equation, self.__y_equation, self.__z_equation)

        self.__figure.canvas.draw()
        plt.show(block = False)

        img = np.fromstring(self.__figure.canvas.tostring_rgb(), dtype = np.uint8, sep = " ")
        img = img.reshape(self.__figure.canvas.get_width_height()[::-1] + (4,))

        return cv2.cvtColor(img, cv2.COLOR_ARGB2BGRA)

if __name__ == "__main__":
    print("Hello")
    graph = Graph()
    graph.graph_type = GraphType.SURFACE
    graph.render()
    image = graph.render_to_image()

    while True:
        cv2.imshow("graph", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

    cv2.destroyAllWindows()
