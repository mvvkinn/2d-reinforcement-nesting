import ezdxf
import io

def convert_flat_dxf_to_stream(flat_string: str):
    tokens = flat_string.strip().split()
    if len(tokens) % 2 != 0:
        raise ValueError("Token length must be even for valid group code/value pairs.")
    lines = [f"{tokens[i]}\n{tokens[i+1]}" for i in range(0, len(tokens), 2)]
    return io.StringIO('\n'.join(lines))

def dxfstr_to_polygon(dxf_str: str):
    stream = convert_flat_dxf_to_stream(dxf_str)
    doc = ezdxf.read(stream)
    doc.audit() # 오류 검사
    return doc