import io
from unittest.mock import Mock, patch

from openpyxl import Workbook

from app.services.file_processor_service import FileProcessorService


def xlsx_bytes():
    workbook = Workbook()
    sales = workbook.active
    sales.title = "Ventas"
    sales.append(["Producto", "Email", "Total", "Total", None])
    sales.append(["Chocolate", "ventas@niilo.co", 1470000, 1500000, "con IVA"])
    sales.append([None, None, None, None, None])
    sales.append(["Panela", None, 1260000, None, None])
    expenses = workbook.create_sheet("Gastos")
    expenses.append(["Concepto", "Valor"])
    expenses.append(["Arriendo", 2500000])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_xlsx_url_is_indexed_row_by_row_with_every_column_and_its_sheet():
    response = Mock(headers={"content-type": "application/octet-stream"})
    response.iter_content.return_value = [xlsx_bytes()]
    with patch("app.services.file_processor_service.requests.get", return_value=response):
        content, metadata = FileProcessorService()._download_and_process_file("https://assets.niilo.co/agents/ventas.xlsx")

    rows = [row for group in content for row in group]
    assert metadata["file_type"] == ".xlsx"
    assert [row["text"] for row in rows] == [
        "Hoja: Ventas | Producto: Chocolate | Email: ventas@niilo.co | Total: 1470000 | Total (4): 1500000 | Columna 5: con IVA",
        "Hoja: Ventas | Producto: Panela | Total: 1260000",
        "Hoja: Gastos | Concepto: Arriendo | Valor: 2500000",
    ]
    assert rows[0]["metadata"] == {
        "Hoja": "Ventas", "Producto": "Chocolate", "Email": "ventas@niilo.co",
        "Total": "1470000", "Total (4)": "1500000", "Columna 5": "con IVA",
    }
