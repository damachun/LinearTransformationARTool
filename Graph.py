from mpl_toolkits import mplot3d
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from enum import Enum
import cv2
from scipy import linalg

class GraphType(Enum):
    LINE      = 0
    WIREFRAME = 1
    SURFACE   = 2
    CONTOUR   = 3

class CoordinateSys(Enum):
    IDENTITY = 0
    ORTHOPRJ = 1
    PRLLPRJ  = 2
    SKEW     = 3

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
        self.__figure.patch.set_alpha(0.0)
        self.__canvas = FigureCanvasAgg(self.__figure)
        if dim3:
            self.__graph = self.__figure.add_subplot(111, projection = "3d")
        else:
            self.__graph = self.__figure.add_subplot()
        self.__graph_type = graph_type

        # coordinate system details
        self.__coordsys  = np.identity(4)
        self.__scale     = np.identity(4)
        self.__rotate    = np.identity(4)
        self.__translate = np.identity(4)

        # axis details
        if apply_equation:
            if dim3:
                x, y = np.meshgrid(x_equation, y_equation)
                self.__x_equation = x
                self.__y_equation = y
                self.__z_equation = z_equation(x, y)
            else:
                size = len(x_equation)
                self.__x_equation = x_equation
                self.__y_equation = y_equation(x_equation)
                self.__z_equation = np.zeros(size, dtype = float)
        else:
            if dim3:
                x, y, z = np.meshgrid(x_equation, y_equation, z_equation)
                self.__x_equation = x
                self.__y_equation = y
                self.__z_equation = z
            else:
                size = len(x_equation)
                self.__x_equation = x_equation
                self.__y_equation = y_equation
                self.__z_equation = np.zeros(size, dtype = float)

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

    def render_helper(self):
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
        self.__canvas.draw()

    def render(self, block = False):
        self.render_helper()
        plt.show(block)

    def render_to_image(self):
        self.render_helper()

        mpl_buffer = self.__canvas.buffer_rgba()
        buffer = np.asarray(mpl_buffer)

        img = np.frombuffer(buffer, dtype = np.uint8)
        img = img.reshape(self.__canvas.get_width_height()[::-1] + (4,))

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        h, w, _ = img.shape
        img_bgra = np.concatenate([img, np.full((h, w, 1), 255, dtype = np.uint8)], axis = -1)

        mask = np.all(img == [255, 255, 255], axis = -1)
        img_bgra[mask, -1] = 0

        cv2.imwrite("graph_image.png", img_bgra)

        return img_bgra

    def scale(self, scale_vec):
        self.__scale = np.identity(4)
        self.__scale[0][0] *= scale_vec[0]
        self.__scale[1][1] *= scale_vec[1]
        self.__scale[2][2] *= scale_vec[2]
    
    def rotate(self, rotate_vec, angle):
        pass
    
    def translate(self, translate_vec):
        pass

    def coordinate_system(self, sys_type = CoordinateSys.IDENTITY, core_vec = np.ones(3)):
        if sys_type == CoordinateSys.IDENTITY:
            self.__coordsys = np.identity(4)
        elif sys_type == CoordinateSys.ORTHOPRJ:
            pass
        elif sys_type == CoordinateSys.PRLLPRJ:
            pass
        elif sys_type == CoordinateSys.SKEW:
            pass

if __name__ == "__main__":
    print("Hello")
    graph = Graph()
    graph.graph_type = GraphType.SURFACE

    bg_image = cv2.cvtColor(cv2.imread("solid_image.jpg"), cv2.COLOR_BGR2BGRA)
    bg_image_resize = cv2.resize(bg_image, (800, 600), interpolation = cv2.INTER_AREA)
    graph_image = graph.render_to_image()
    graph_image_resize = cv2.resize(graph_image, (800, 600), interpolation = cv2.INTER_AREA)

    def blending(bg_image, fg_image):
        fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGRA2GRAY)
        _, fg_graymask = cv2.threshold(fg_gray, 120, 255, cv2.THRESH_BINARY)
        bg_or = cv2.bitwise_or(bg_image, bg_image, mask = fg_graymask)
        fg_graymaskinv = cv2.bitwise_not(fg_gray)
        fg_maskinv = cv2.bitwise_and(fg_image, fg_image, mask = fg_graymaskinv)
        return cv2.add(bg_or, fg_maskinv)

    image = blending(bg_image_resize, graph_image_resize)

    while True:
        cv2.imshow("transparent graph", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

    cv2.destroyAllWindows()
