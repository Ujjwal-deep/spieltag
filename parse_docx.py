import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text = []
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            for paragraph in tree.findall('.//w:p', namespaces):
                para_text = "".join(node.text for node in paragraph.findall('.//w:t', namespaces) if node.text)
                if para_text:
                    text.append(para_text)
        return "\n".join(text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = extract_text_from_docx(sys.argv[1])
        with open("blueprint_parsed.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Done")
