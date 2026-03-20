from langgraph.graph import StateGraph, END

from src.state import FormGuideState
from src.node.preprocess import preprocess_node
from src.node.detect_boxes import detect_boxes_node
from src.node.classify_fields import classify_fields_node
from src.node.draw_boxes import draw_boxes_node
from src.node.finalize import finalize_node


def build_graph():
    graph = StateGraph(FormGuideState)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("detect_boxes", detect_boxes_node)
    graph.add_node("classify_fields", classify_fields_node)
    graph.add_node("draw_boxes", draw_boxes_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "detect_boxes")
    graph.add_edge("detect_boxes", "classify_fields")
    graph.add_edge("classify_fields", "draw_boxes")
    graph.add_edge("draw_boxes", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
