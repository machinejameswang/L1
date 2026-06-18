from manim import *

class SupportVectorsScene(Scene):
    def construct(self):
        plane = NumberPlane(x_range=(-4, 4, 1), y_range=(-3, 3, 1))
        self.add(plane)

        # Draw points
        group1 = VGroup(*[Dot(plane.c2p(x, y), color=RED) for x, y in [(-2, 1), (-1, 2), (-3, 0), (-2.5, 2.5)]])
        group2 = VGroup(*[Dot(plane.c2p(x, y), color=BLUE) for x, y in [(1, -1), (2, -2), (3, -0.5), (2.5, -2.5)]])
        self.play(FadeIn(group1), FadeIn(group2))

        # Show decision boundary and margins
        best_line = DashedLine(plane.c2p(-3, -3), plane.c2p(3, 3), color=WHITE)
        margin1 = Line(plane.c2p(-3, -1), plane.c2p(1, 3), color=YELLOW, stroke_width=2)
        margin2 = Line(plane.c2p(-1, -3), plane.c2p(3, 1), color=YELLOW, stroke_width=2)
        self.play(Create(best_line), Create(margin1), Create(margin2))

        # Highlight SVs
        sv_g1 = group1[0]
        sv_g2 = group2[0]
        marks = VGroup(
            Circle(color=YELLOW).surround(sv_g1),
            Circle(color=YELLOW).surround(sv_g2)
        )
        self.play(Create(marks))
        
        note = Text("These are Support Vectors", font_size=24, color=YELLOW).next_to(sv_g1, UP)
        self.play(Write(note))
        self.wait(1)

        # Move non-SV point, show boundary doesn't change
        non_sv = group1[1]
        self.play(non_sv.animate.move_to(plane.c2p(-1.5, 2.5)), run_time=1.5)
        
        note2 = Text("Moving other points doesn't affect the boundary!", font_size=24).to_edge(DOWN)
        self.play(Write(note2))
        self.wait(2)
