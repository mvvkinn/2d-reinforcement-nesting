import ezdxf
import io

def convert_flat_dxf_to_stream(flat_string: str):
    try:
        # 먼저 일반적인 방식으로 토큰 분리
        tokens = flat_string.strip().split()
        if len(tokens) % 2 != 0:
            # 토큰 개수가 홀수인 경우, 마지막 토큰을 제거하여 짝수로 만듦
            print(f"경고: 토큰 개수가 홀수입니다 ({len(tokens)}개). 마지막 토큰을 제거합니다.")
            tokens = tokens[:-1]

        # 그룹 코드와 값 쌍으로 변환
        lines = [f"{tokens[i]}\n{tokens[i+1]}" for i in range(0, len(tokens), 2)]
        return io.StringIO('\n'.join(lines))
    except Exception as e:
        # 변환 실패 시 기본 DXF 문자열 반환
        print(f"DXF 문자열 변환 중 오류 발생: {str(e)}")
        # 최소한의 유효한 DXF 문자열 생성
        minimal_dxf = "0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF"
        return io.StringIO(minimal_dxf)

def dxfstr_to_polygon(dxf_str: str):
    try:
        stream = convert_flat_dxf_to_stream(dxf_str)
        doc = ezdxf.read(stream)
        doc.audit() # 오류 검사
        return doc
    except Exception as e:
        print(f"DXF 문서 읽기 중 오류 발생: {str(e)}")
        # 빈 DXF 문서 반환
        try:
            doc = ezdxf.new()
            return doc
        except:
            return None
