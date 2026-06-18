from manim import *
import numpy as np

class KernelTrickScene(ThreeDScene):
    def construct(self):
        title = Text("Kernel Trick helps SVM separate non-linear data", font_size=32).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        axes = ThreeDAxes(
            x_range=(-2.5, 2.5, 1),
            y_range=(-2.5, 2.5, 1),
            z_range=(0, 1.5, 0.5),
            x_length=6, y_length=6, z_length=3
        )
        self.play(FadeIn(axes))

        # 2D concentric circles
        inner = [axes.c2p(0.5*np.cos(a), 0.5*np.sin(a), 0) for a in np.linspace(0, 2*np.pi, 12, endpoint=False)]
        outer = [axes.c2p(1.5*np.cos(a), 1.5*np.sin(a), 0) for a in np.linspace(0, 2*np.pi, 24, endpoint=False)]
        
        inner_dots = VGroup(*[Dot(p, color=BLUE) for p in inner])
        outer_dots = VGroup(*[Dot(p, color=RED) for p in outer])
        self.play(FadeIn(inner_dots), FadeIn(outer_dots))

        note = Text("Linearly inseparable in 2D", font_size=24, color=RED).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note)
        self.play(Write(note))
        self.wait(1)
        self.play(FadeOut(note))

        # Kernel lift function: z = exp(-gamma * (x^2 + y^2))
        gamma = 1.0
        def lift(p):
            x, y, z = axes.p2c(p)
            z_new = np.exp(-gamma * (x**2 + y**2))
            return axes.c2p(x, y, z_new)

        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)

        lifted_inner = VGroup(*[Dot(lift(p.get_center()), color=BLUE) for p in inner_dots])
        lifted_outer = VGroup(*[Dot(lift(p.get_center()), color=RED) for p in outer_dots])
        
        self.play(
            ReplacementTransform(inner_dots, lifted_inner),
            ReplacementTransform(outer_dots, lifted_outer),
            run_time=2
        )

        plane = Surface(
            lambda u, v: axes.c2p(u, v, 0.4),
            u_range=(-2.5, 2.5), v_range=(-2.5, 2.5),
            resolution=(2, 2),
            fill_opacity=0.3,
            color=WHITE
        )
        self.play(FadeIn(plane))
        
        note2 = Text("Separable in 3D!", font_size=24, color=GREEN).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note2)
        self.play(Write(note2))
        
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
