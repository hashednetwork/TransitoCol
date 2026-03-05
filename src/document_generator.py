"""
Derecho de Petición Generator for Colombian Traffic Fines
Generates PDF documents based on user case information
"""
import os
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_RIGHT


class DerechoPeticionGenerator:
    """Generate Derecho de Petición PDF documents."""
    
    # Document templates for different case types
    TEMPLATES = {
        "prescripcion": {
            "title": "DERECHO DE PETICIÓN - PRESCRIPCIÓN DE MULTA",
            "legal_basis": """
Fundamento mi solicitud en los siguientes hechos y normas:

1. El artículo 23 de la Constitución Política de Colombia consagra el derecho fundamental de petición.

2. El artículo 159 de la Ley 769 de 2002 (Código Nacional de Tránsito) establece que la acción para cobrar multas por infracciones de tránsito prescribe en tres (3) años contados a partir de la fecha de la comisión del hecho.

3. El artículo 2536 del Código Civil establece que la acción ejecutiva se prescribe por cinco (5) años y la ordinaria por diez (10) años, pero tratándose de multas de tránsito aplica el término especial de tres (3) años.

4. La Corte Constitucional en múltiples sentencias ha reiterado que el Estado no puede cobrar indefinidamente sanciones pecuniarias, debiendo respetar los términos de prescripción establecidos en la ley.
""",
            "request": """
PETICIÓN:

Con fundamento en lo anteriormente expuesto, respetuosamente solicito:

PRIMERO: Declarar la PRESCRIPCIÓN del comparendo/multa identificado anteriormente, por haber transcurrido más de tres (3) años desde la fecha de la presunta infracción sin que se haya hecho efectivo su cobro.

SEGUNDO: Ordenar el RETIRO de dicha multa del sistema SIMIT y de cualquier otro registro que afecte mi historial como conductor.

TERCERO: Expedir certificación donde conste que no tengo obligaciones pendientes respecto a esta multa.
"""
        },
        
        "fotomulta_notificacion": {
            "title": "DERECHO DE PETICIÓN - NULIDAD FOTOMULTA POR FALTA DE NOTIFICACIÓN",
            "legal_basis": """
Fundamento mi solicitud en los siguientes hechos y normas:

1. El artículo 23 de la Constitución Política de Colombia consagra el derecho fundamental de petición.

2. El artículo 29 de la Constitución Política garantiza el debido proceso, el cual incluye el derecho a ser notificado oportunamente de las actuaciones que puedan afectar los derechos del ciudadano.

3. La Ley 1843 de 2017 en su artículo 8 establece que la notificación del comparendo debe realizarse dentro de los tres (3) días hábiles siguientes a la fecha de la presunta infracción.

4. La Corte Constitucional en Sentencia C-038 de 2020 reiteró que la notificación tardía o la falta de notificación vulnera el derecho al debido proceso y genera la nulidad del comparendo.

5. El Decreto 2106 de 2019 establece que las autoridades deben garantizar el derecho a la defensa mediante notificación oportuna.
""",
            "request": """
PETICIÓN:

Con fundamento en lo anteriormente expuesto, respetuosamente solicito:

PRIMERO: Declarar la NULIDAD del foto comparendo identificado anteriormente, por violación al debido proceso ante la falta de notificación oportuna.

SEGUNDO: Ordenar el ARCHIVO DEFINITIVO del proceso contravencional.

TERCERO: Ordenar el RETIRO de dicha multa del sistema SIMIT y de cualquier otro registro.

CUARTO: En caso de existir cualquier cobro o embargo derivado de esta multa, ordenar su levantamiento inmediato.
"""
        },
        
        "fotomulta_identificacion": {
            "title": "DERECHO DE PETICIÓN - NULIDAD FOTOMULTA POR FALTA DE IDENTIFICACIÓN DEL CONDUCTOR",
            "legal_basis": """
Fundamento mi solicitud en los siguientes hechos y normas:

1. El artículo 23 de la Constitución Política de Colombia consagra el derecho fundamental de petición.

2. El artículo 29 de la Constitución Política garantiza la presunción de inocencia y el debido proceso.

3. La Corte Constitucional en Sentencia C-038 de 2020 estableció que en materia de fotomultas, la responsabilidad NO puede trasladarse automáticamente al propietario del vehículo cuando no se identifica plenamente al conductor infractor.

4. El artículo 129 de la Ley 769 de 2002 establece que las sanciones por infracciones de tránsito son de carácter personal.

5. La presunción de inocencia implica que la autoridad debe probar quién cometió la infracción, no puede presumirse que fue el propietario.
""",
            "request": """
PETICIÓN:

Con fundamento en lo anteriormente expuesto, respetuosamente solicito:

PRIMERO: Declarar la NULIDAD del foto comparendo identificado anteriormente, por cuanto NO se identifica claramente al conductor que presuntamente cometió la infracción.

SEGUNDO: Exonerar de cualquier responsabilidad al suscrito como propietario del vehículo, dado que la responsabilidad en materia de tránsito es personal.

TERCERO: Ordenar el RETIRO de dicha multa del sistema SIMIT y de cualquier otro registro.

CUARTO: Archivar definitivamente el presente proceso.
"""
        },
        

        "parqueo_señalizacion": {
            "title": "DERECHO DE PETICIÓN - NULIDAD COMPARENDO POR PARQUEO EN ZONA SIN SEÑALIZACIÓN (LEY 2252/2022)",
            "legal_basis": """
Fundamento mi solicitud en los siguientes hechos y normas:

1. El artículo 23 de la Constitución Política de Colombia consagra el derecho fundamental de petición.

2. El artículo 29 de la Constitución Política garantiza el debido proceso y la presunción de inocencia.

3. La Ley 2252 de 2022 modificó el artículo 112 de la Ley 769 de 2002 (Código Nacional de Tránsito Terrestre), estableciendo en su Parágrafo 1° que las autoridades de tránsito SOLO podrán imponer sanciones o comparendos por estacionamiento en zonas prohibidas cuando dichas zonas estén debidamente señalizadas conforme al Manual de Señalización Vial vigente expedido por el Ministerio de Transporte.

4. El Parágrafo 1° de la Ley 2252 de 2022 expresamente prohíbe imponer comparendos en zonas no señalizadas o con señalización ilegible, deteriorada o que no cumpla los estándares técnicos exigidos.

5. El Parágrafo 3° de la Ley 2252 de 2022 establece que los dispositivos tecnológicos de control de tránsito, incluidas las cámaras de fotodetección, solo podrán registrar y sancionar infracciones de estacionamiento en zonas con señalización vigente, debidamente instalada y en perfectas condiciones de visibilidad. Las multas derivadas de zonas sin señalización adecuada son NULAS DE PLENO DERECHO.

6. El Manual de Señalización Vial vigente (Resolución 20243040045005 de 2024) establece los estándares técnicos que debe cumplir toda señalización vial para ser jurídicamente válida.

7. La Superintendencia de Transporte y las autoridades competentes están obligadas a garantizar que la señalización cumpla los requisitos técnicos antes de proceder a cualquier sanción.
""",
            "request": """
PETICIÓN:

Con fundamento en lo anteriormente expuesto, respetuosamente solicito:

PRIMERO: Declarar la NULIDAD de pleno derecho del comparendo/multa identificado anteriormente, por cuanto la zona donde se impuso la sanción NO contaba con señalización debida conforme a lo exigido por la Ley 2252 de 2022 y el Manual de Señalización Vial vigente.

SEGUNDO: Aportar prueba documental y fotográfica de:
   a) La señalización existente en la zona al momento de la infracción.
   b) Que dicha señalización cumple los estándares técnicos del Manual de Señalización Vial.
   c) Fecha de instalación y última revisión de la señal.

TERCERO: En caso de no poder acreditar la señalización adecuada, ordenar el ARCHIVO DEFINITIVO del proceso y el RETIRO de la multa del sistema SIMIT y cualquier otro registro.

CUARTO: Levantar de manera inmediata cualquier restricción, embargo o anotación derivada de esta multa que afecte mi licencia de conducción o el vehículo.

QUINTO: Expedir certificación donde conste que no tengo obligaciones pendientes respecto a esta multa.
"""
        },
        "fotomulta_señalizacion": {
            "title": "DERECHO DE PETICIÓN - NULIDAD FOTOMULTA POR FALTA DE SEÑALIZACIÓN",
            "legal_basis": """
Fundamento mi solicitud en los siguientes hechos y normas:

1. El artículo 23 de la Constitución Política de Colombia consagra el derecho fundamental de petición.

2. La Ley 1843 de 2017 en su artículo 2 establece que todo sistema de detección de infracciones debe cumplir con criterios técnicos de seguridad vial para su instalación y operación.

3. El artículo 5 de la Ley 1843 de 2017 establece que debe existir señalización preventiva e informativa que indique la presencia de sistemas de fotodetección, ubicada al menos 500 metros antes del lugar de detección.

4. El Decreto 2106 de 2019, artículo 109, establece que los sistemas de fotomultas deben contar con autorización de la Agencia Nacional de Seguridad Vial.

5. La Superintendencia de Transporte ha sancionado a múltiples secretarías de tránsito por operar cámaras sin la señalización requerida.
""",
            "request": """
PETICIÓN:

Con fundamento en lo anteriormente expuesto, respetuosamente solicito:

PRIMERO: Declarar la NULIDAD del foto comparendo identificado anteriormente, por incumplimiento de los requisitos de señalización establecidos en la Ley 1843 de 2017.

SEGUNDO: Solicito se aporte prueba de:
   a) Autorización vigente de la Agencia Nacional de Seguridad Vial para el sistema de fotodetección.
   b) Señalización instalada a 500 metros del punto de detección.
   c) Cumplimiento de todos los requisitos técnicos establecidos.

TERCERO: Ordenar el RETIRO de dicha multa del sistema SIMIT y de cualquier otro registro.

CUARTO: Archivar definitivamente el presente proceso.
"""
        }
    }
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Configure custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Justified',
            parent=self.styles['Normal'],
            alignment=TA_JUSTIFY,
            fontSize=11,
            leading=14,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name='Header',
            parent=self.styles['Heading1'],
            alignment=TA_CENTER,
            fontSize=14,
            spaceAfter=20
        ))
        self.styles.add(ParagraphStyle(
            name='RightAlign',
            parent=self.styles['Normal'],
            alignment=TA_RIGHT,
            fontSize=11
        ))
    
    def generate_document(
        self,
        template_type: str,
        nombre_completo: str,
        cedula: str,
        direccion: str,
        telefono: str,
        email: str,
        ciudad_autoridad: str,
        numero_comparendo: str,
        fecha_infraccion: str,
        placa_vehiculo: str,
        hechos_adicionales: str = ""
    ) -> BytesIO:
        """
        Generate a Derecho de Petición PDF document.
        
        Args:
            template_type: Type of template (prescripcion, fotomulta_notificacion, etc.)
            nombre_completo: Full name of the petitioner
            cedula: ID number
            direccion: Address
            telefono: Phone number
            email: Email address
            ciudad_autoridad: City of the transit authority
            numero_comparendo: Fine/ticket number
            fecha_infraccion: Date of the alleged infraction
            placa_vehiculo: Vehicle plate number
            hechos_adicionales: Additional facts specific to the case
            
        Returns:
            BytesIO buffer containing the PDF
        """
        if template_type not in self.TEMPLATES:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template = self.TEMPLATES[template_type]
        buffer = BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        story = []
        fecha_actual = datetime.now().strftime("%d de %B de %Y").replace(
            "January", "enero").replace("February", "febrero").replace(
            "March", "marzo").replace("April", "abril").replace(
            "May", "mayo").replace("June", "junio").replace(
            "July", "julio").replace("August", "agosto").replace(
            "September", "septiembre").replace("October", "octubre").replace(
            "November", "noviembre").replace("December", "diciembre")
        
        # Header with date and city
        story.append(Paragraph(f"{ciudad_autoridad}, {fecha_actual}", self.styles['RightAlign']))
        story.append(Spacer(1, 20))
        
        # Addressee
        story.append(Paragraph("Señores", self.styles['Normal']))
        story.append(Paragraph(f"<b>SECRETARÍA DE TRÁNSITO Y TRANSPORTE DE {ciudad_autoridad.upper()}</b>", self.styles['Normal']))
        story.append(Paragraph("Ciudad", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Subject
        story.append(Paragraph(f"<b>Asunto: {template['title']}</b>", self.styles['Normal']))
        story.append(Paragraph(f"<b>Comparendo No.: {numero_comparendo}</b>", self.styles['Normal']))
        story.append(Paragraph(f"<b>Placa: {placa_vehiculo}</b>", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Greeting
        story.append(Paragraph("Respetados señores:", self.styles['Justified']))
        story.append(Spacer(1, 10))
        
        # Introduction
        intro = f"""
        Yo, <b>{nombre_completo}</b>, identificado(a) con cédula de ciudadanía número <b>{cedula}</b>, 
        en ejercicio del derecho fundamental de petición consagrado en el artículo 23 de la Constitución 
        Política de Colombia, reglamentado por la Ley 1755 de 2015, me permito presentar la siguiente 
        solicitud relacionada con el comparendo número <b>{numero_comparendo}</b>, presuntamente impuesto 
        el día <b>{fecha_infraccion}</b> al vehículo de placas <b>{placa_vehiculo}</b>.
        """
        story.append(Paragraph(intro, self.styles['Justified']))
        story.append(Spacer(1, 15))
        
        # Facts section
        story.append(Paragraph("<b>HECHOS:</b>", self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        if hechos_adicionales:
            story.append(Paragraph(hechos_adicionales, self.styles['Justified']))
            story.append(Spacer(1, 10))
        
        # Legal basis
        story.append(Paragraph("<b>FUNDAMENTOS DE DERECHO:</b>", self.styles['Normal']))
        for para in template['legal_basis'].strip().split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['Justified']))
        story.append(Spacer(1, 15))
        
        # Request
        for para in template['request'].strip().split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['Justified']))
        story.append(Spacer(1, 15))
        
        # Notifications
        story.append(Paragraph("<b>NOTIFICACIONES:</b>", self.styles['Normal']))
        story.append(Spacer(1, 5))
        notificaciones = f"""
        Recibiré notificaciones en:<br/>
        <b>Dirección:</b> {direccion}<br/>
        <b>Teléfono:</b> {telefono}<br/>
        <b>Correo electrónico:</b> {email}
        """
        story.append(Paragraph(notificaciones, self.styles['Justified']))
        story.append(Spacer(1, 20))
        
        # Closing
        story.append(Paragraph("Agradezco su atención y quedo atento(a) a su respuesta dentro de los términos de ley (15 días hábiles).", self.styles['Justified']))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Atentamente,", self.styles['Normal']))
        story.append(Spacer(1, 40))
        
        story.append(Paragraph("_" * 40, self.styles['Normal']))
        story.append(Paragraph(f"<b>{nombre_completo}</b>", self.styles['Normal']))
        story.append(Paragraph(f"C.C. {cedula}", self.styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def get_available_templates(self) -> dict:
        """Return available template types with descriptions."""
        return {
            "prescripcion": "Multa con más de 3 años de antigüedad (prescripción)",
            "fotomulta_notificacion": "Fotomulta sin notificación oportuna (más de 3 días)",
            "fotomulta_identificacion": "Fotomulta donde no se identifica al conductor",
            "fotomulta_señalizacion": "Fotomulta sin señalización adecuada (500m antes)"
        }


# Test function
if __name__ == "__main__":
    gen = DerechoPeticionGenerator()
    
    # Test generate a document
    pdf = gen.generate_document(
        template_type="prescripcion",
        nombre_completo="Juan Carlos Pérez García",
        cedula="1.234.567.890",
        direccion="Calle 123 # 45-67, Bogotá",
        telefono="300 123 4567",
        email="juan.perez@email.com",
        ciudad_autoridad="Bogotá D.C.",
        numero_comparendo="ABC123456789",
        fecha_infraccion="15 de enero de 2022",
        placa_vehiculo="ABC-123",
        hechos_adicionales="El comparendo fue impuesto hace más de 3 años y nunca recibí notificación alguna. No he sido citado a audiencia ni se ha iniciado proceso de cobro coactivo."
    )
    
    # Save test PDF
    with open("test_derecho_peticion.pdf", "wb") as f:
        f.write(pdf.read())
    print("Test PDF generated: test_derecho_peticion.pdf")
