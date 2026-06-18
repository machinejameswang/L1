from manim import *

class SVMMarginScene(Scene):
    def construct(self):
        title = Text("SVM chooses the hyperplane with the maximum margin", font_size=32).to_edge(UP)
        self.play(Write(title))

        plane = NumberPlane(x_range=(-4, 4, 1), y_range=(-3, 3, 1))
        self.play(FadeIn(plane))

        # Two distinct groups
        group1 = VGroup(*[Dot(plane.c2p(x, y), color=RED) for x, y in [(-2, 1), (-1, 2), (-3, 0), (-1.5, 0.5)]])
        group2 = VGroup(*[Dot(plane.c2p(x, y), color=BLUE) for x, y in [(1, -1), (2, -2), (3, -0.5), (1.5, -2)]])
        self.play(FadeIn(group1), FadeIn(group2))

        # Show multiple bad lines
        l1 = Line(plane.c2p(-1, -3), plane.c2p(1, 3), color=GRAY)
        l2 = Line(plane.c2p(-3, -1), plane.c2p(3, 1), color=GRAY)
        self.play(Create(l1))
        self.wait(1)
        self.play(ReplacementTransform(l1, l2))
        self.wait(1)
        self.play(FadeOut(l2))

        # Show best line
        best_line = DashedLine(plane.c2p(-3, -3), plane.c2p(3, 3), color=WHITE)
        margin1 = Line(plane.c2p(-3, -1), plane.c2p(1, 3), color=YELLOW, stroke_width=2)
        margin2 = Line(plane.c2p(-1, -3), plane.c2p(3, 1), color=YELLOW, stroke_width=2)
        self.play(Create(best_line))
        self.play(Create(margin1), Create(margin2))

        # Highlight support vectors
        sv_marks = VGroup(
            Circle(color=YELLOW).surround(group1[0]),
            Circle(color=YELLOW).surround(group2[0])
        )
        self.play(Create(sv_marks))
        self.wait(3)
