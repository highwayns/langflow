import re
import pytesseract
import pdf2image
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.document_loaders.pdf import BasePDFLoader
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Union
from langchain.document_loaders.parsers.pdf import (
    AmazonTextractPDFParser,
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
)


class PyPDFLoaderWithOCR(BasePDFLoader):
    """Load `PDF using `pypdf` and chunks at character level.

    Loader also stores page numbers in metadata.
    """

    def __init__(
        self, file_path: str, password: Optional[Union[str, bytes]] = None
    ) -> None:
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        self.parser = PyPDFParser(password=password)
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return self.extract_text_with_ocr(self.file_path)

    def extract_text_with_ocr(self, pdf_path):
        pdf_reader = PdfReader(open(pdf_path, 'rb'))
        total_pages = len(pdf_reader.pages)

        images = self.to_images(pdf_path, 1, total_pages)
        
        documents = [self.normalize(self.to_string(image), pdf_path) for image in images]

        return documents

    def to_images(self, pdf_path: str, first_page: int = None, last_page: int = None) -> list:
        """ Convert a PDF to a PNG image.

        Args:
            pdf_path (str): PDF path
            first_page (int): First page starting 1 to be converted
            last_page (int): Last page to be converted

        Returns:
            list: List of image data
        """

        print(f'Convert a PDF ({pdf_path}) to a png...')
        images = pdf2image.convert_from_path(
            pdf_path=pdf_path,
            fmt='png',
            first_page=first_page,
            last_page=last_page,
        )
        print(f'A total of converted png images is {len(images)}.')
        return images


    def to_string(self, image) -> str:
        """ OCR an image data.

        Args:
            image: Image data

        Returns:
            str: OCR processed characters
        """

        print(f'Extract characters from an image...')
        return pytesseract.image_to_string(image, lang='jpn')


    def normalize(self, target: str, pdf_path: str) -> str:
        """ Normalize result text.

        Applying the following:
        - Remove new line.
        - Remove spaces between Japanese characters.

        Args:
            target (str): Target text to be normalized

        Returns:
            str: Normalized text
        """

        result = re.sub('\n', '', target)
        result = re.sub('([あ-んア-ン一-鿐])\s+((?=[あ-んア-ン一-鿐]))', r'\1\2', result)
        doc = Document(
                page_content=result,
                metadata={
                    "source": pdf_path,
                })
        return doc
