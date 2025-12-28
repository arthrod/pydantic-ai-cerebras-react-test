"""Shared legal document corpus for all config modules.

Contains:
- 11 Brazilian law documents
- 11 American law documents

Document types:
- 8 contracts (4 BR, 4 US)
- 8 court briefs (4 BR, 4 US)
- 6 legal opinions (3 BR, 3 US)
"""

from typing import Any

CORPUS: list[dict[str, Any]] = [
    # === BRAZILIAN CONTRACTS (4) ===
    {
        "id": "BR-CT-001",
        "title": "Contrato de Locação Comercial",
        "snippet": "Contrato de locação de imóvel comercial em São Paulo.",
        "doc_type": "contract",
        "jurisdiction": "brazil",
        "tags": ["lease", "commercial", "real-estate", "sao-paulo"],
        "content": "Este contrato de locação comercial estabelece os termos para aluguel de imóvel na Av. Paulista, incluindo valor do aluguel, prazo de 5 anos, e cláusulas de reajuste pelo IGPM.",
    },
    {
        "id": "BR-CT-002",
        "title": "Contrato de Trabalho CLT",
        "snippet": "Contrato de trabalho conforme CLT com benefícios.",
        "doc_type": "contract",
        "jurisdiction": "brazil",
        "tags": ["employment", "clt", "labor", "benefits"],
        "content": "Contrato de trabalho por prazo indeterminado conforme a CLT. Inclui salário base, vale-transporte, vale-refeição, plano de saúde, e jornada de 44 horas semanais.",
    },
    {
        "id": "BR-CT-003",
        "title": "Contrato de Prestação de Serviços",
        "snippet": "Contrato de prestação de serviços de consultoria empresarial.",
        "doc_type": "contract",
        "jurisdiction": "brazil",
        "tags": ["services", "consulting", "business", "pj"],
        "content": "Contrato de prestação de serviços entre pessoa jurídica e empresa contratante. Define escopo de consultoria, honorários mensais, prazo de 12 meses, e cláusula de confidencialidade.",
    },
    {
        "id": "BR-CT-004",
        "title": "Contrato de Compra e Venda de Imóvel",
        "snippet": "Contrato de compra e venda de apartamento residencial.",
        "doc_type": "contract",
        "jurisdiction": "brazil",
        "tags": ["sale", "purchase", "real-estate", "residential"],
        "content": "Contrato de compra e venda de imóvel residencial no Rio de Janeiro. Valor de R$ 850.000, financiamento pela Caixa Econômica Federal, escritura definitiva após quitação.",
    },
    # === AMERICAN CONTRACTS (4) ===
    {
        "id": "US-CT-001",
        "title": "Commercial Lease Agreement",
        "snippet": "Commercial property lease agreement for office space in Manhattan.",
        "doc_type": "contract",
        "jurisdiction": "usa",
        "tags": ["lease", "commercial", "real-estate", "new-york"],
        "content": "This Commercial Lease Agreement governs the rental of office space at 350 Fifth Avenue, NYC. Terms include base rent of $75/sqft, 10-year term, triple net lease structure, and annual 3% escalation.",
    },
    {
        "id": "US-CT-002",
        "title": "Employment Agreement",
        "snippet": "Executive employment agreement with stock options.",
        "doc_type": "contract",
        "jurisdiction": "usa",
        "tags": ["employment", "executive", "stock-options", "compensation"],
        "content": "Employment Agreement for Chief Technology Officer position. Includes base salary of $350,000, performance bonus up to 40%, stock options vesting over 4 years, and 12-month non-compete clause.",
    },
    {
        "id": "US-CT-003",
        "title": "Master Services Agreement",
        "snippet": "MSA for software development and IT consulting services.",
        "doc_type": "contract",
        "jurisdiction": "usa",
        "tags": ["services", "software", "consulting", "msa"],
        "content": "Master Services Agreement for ongoing software development. Covers hourly rates, SOW process, IP ownership, liability caps, indemnification provisions, and Delaware choice of law.",
    },
    {
        "id": "US-CT-004",
        "title": "Asset Purchase Agreement",
        "snippet": "Agreement for acquisition of business assets and goodwill.",
        "doc_type": "contract",
        "jurisdiction": "usa",
        "tags": ["acquisition", "purchase", "assets", "m&a"],
        "content": "Asset Purchase Agreement for acquisition of target company's assets. Purchase price of $12M, includes inventory, equipment, customer lists, and goodwill. Subject to Hart-Scott-Rodino filing.",
    },
    # === BRAZILIAN COURT BRIEFS (4) ===
    {
        "id": "BR-CB-001",
        "title": "Petição Inicial - Ação Trabalhista",
        "snippet": "Petição inicial de reclamação trabalhista por verbas rescisórias.",
        "doc_type": "brief",
        "jurisdiction": "brazil",
        "tags": ["labor", "termination", "severance", "trt"],
        "content": "Petição inicial perante a Vara do Trabalho pleiteando verbas rescisórias não pagas: aviso prévio, 13º proporcional, férias + 1/3, FGTS + multa de 40%. Valor da causa: R$ 45.000.",
    },
    {
        "id": "BR-CB-002",
        "title": "Contestação - Ação de Despejo",
        "snippet": "Contestação em ação de despejo por falta de pagamento.",
        "doc_type": "brief",
        "jurisdiction": "brazil",
        "tags": ["eviction", "rent", "defense", "civil"],
        "content": "Contestação à ação de despejo alegando exceção de contrato não cumprido. Locador não realizou reparos essenciais no imóvel conforme art. 22 da Lei 8.245/91. Requer improcedência.",
    },
    {
        "id": "BR-CB-003",
        "title": "Recurso de Apelação Cível",
        "snippet": "Apelação contra sentença em ação de indenização.",
        "doc_type": "brief",
        "jurisdiction": "brazil",
        "tags": ["appeal", "damages", "civil", "tjsp"],
        "content": "Recurso de Apelação ao TJSP contra sentença que julgou improcedente pedido de indenização por danos morais. Alega erro na valoração da prova e pede reforma para condenar réu em R$ 50.000.",
    },
    {
        "id": "BR-CB-004",
        "title": "Mandado de Segurança Tributário",
        "snippet": "MS contra cobrança indevida de ICMS em operação interestadual.",
        "doc_type": "brief",
        "jurisdiction": "brazil",
        "tags": ["tax", "icms", "mandamus", "constitutional"],
        "content": "Mandado de Segurança contra ato do Secretário da Fazenda que exigiu diferencial de alíquota de ICMS em operação já tributada na origem. Direito líquido e certo à não bitributação.",
    },
    # === AMERICAN COURT BRIEFS (4) ===
    {
        "id": "US-CB-001",
        "title": "Motion to Dismiss - Contract Dispute",
        "snippet": "Motion to dismiss breach of contract claim under FRCP 12(b)(6).",
        "doc_type": "brief",
        "jurisdiction": "usa",
        "tags": ["motion", "dismiss", "contract", "federal"],
        "content": "Motion to Dismiss pursuant to FRCP 12(b)(6) for failure to state a claim. Plaintiff's breach of contract claim fails because the alleged agreement lacks essential terms and consideration. Iqbal/Twombly standard not met.",
    },
    {
        "id": "US-CB-002",
        "title": "Opposition to Summary Judgment",
        "snippet": "Opposition brief in employment discrimination case.",
        "doc_type": "brief",
        "jurisdiction": "usa",
        "tags": ["employment", "discrimination", "summary-judgment", "title-vii"],
        "content": "Opposition to Defendant's Motion for Summary Judgment in Title VII discrimination case. Plaintiff presents genuine dispute of material fact regarding pretext. McDonnell Douglas burden-shifting framework satisfied.",
    },
    {
        "id": "US-CB-003",
        "title": "Appellate Brief - Ninth Circuit",
        "snippet": "Opening brief on appeal challenging preliminary injunction.",
        "doc_type": "brief",
        "jurisdiction": "usa",
        "tags": ["appeal", "injunction", "ninth-circuit", "ip"],
        "content": "Opening Brief in the Ninth Circuit Court of Appeals. District court abused discretion in denying preliminary injunction. Appellant demonstrates likelihood of success on trademark infringement claim under Lanham Act.",
    },
    {
        "id": "US-CB-004",
        "title": "Class Certification Motion",
        "snippet": "Motion for class certification in consumer protection action.",
        "doc_type": "brief",
        "jurisdiction": "usa",
        "tags": ["class-action", "consumer", "certification", "frcp-23"],
        "content": "Motion for Class Certification under FRCP 23(b)(3). Proposed class of 50,000 consumers meets numerosity, commonality, typicality, and adequacy requirements. Common questions of law predominate.",
    },
    # === BRAZILIAN LEGAL OPINIONS (3) ===
    {
        "id": "BR-OP-001",
        "title": "Parecer Jurídico - Fusão de Empresas",
        "snippet": "Parecer sobre aspectos societários de fusão entre S.A.s.",
        "doc_type": "opinion",
        "jurisdiction": "brazil",
        "tags": ["merger", "corporate", "cvm", "antitrust"],
        "content": "Parecer jurídico analisando a fusão entre duas sociedades anônimas. Aborda aprovação em AGE, direito de retirada dos dissidentes, notificação ao CADE, e registro na CVM. Operação viável juridicamente.",
    },
    {
        "id": "BR-OP-002",
        "title": "Parecer - Compliance Anticorrupção",
        "snippet": "Análise de programa de compliance conforme Lei 12.846/13.",
        "doc_type": "opinion",
        "jurisdiction": "brazil",
        "tags": ["compliance", "anticorruption", "corporate", "lgpd"],
        "content": "Parecer sobre adequação do programa de integridade à Lei Anticorrupção e LGPD. Recomenda melhorias no canal de denúncias, due diligence de terceiros, e treinamentos periódicos.",
    },
    {
        "id": "BR-OP-003",
        "title": "Parecer Tributário - Reforma",
        "snippet": "Análise dos impactos da reforma tributária no setor de serviços.",
        "doc_type": "opinion",
        "jurisdiction": "brazil",
        "tags": ["tax", "reform", "services", "ibs"],
        "content": "Parecer técnico sobre impactos da reforma tributária (IBS/CBS) em empresas de serviços. Analisa período de transição, créditos, e estratégias de planejamento tributário lícito.",
    },
    # === AMERICAN LEGAL OPINIONS (3) ===
    {
        "id": "US-OP-001",
        "title": "Legal Opinion - M&A Due Diligence",
        "snippet": "Opinion letter on target company's legal compliance status.",
        "doc_type": "opinion",
        "jurisdiction": "usa",
        "tags": ["m&a", "due-diligence", "corporate", "compliance"],
        "content": "Legal opinion letter for acquiring company regarding target's material contracts, litigation exposure, IP portfolio, and regulatory compliance. No material legal impediments to closing identified.",
    },
    {
        "id": "US-OP-002",
        "title": "SEC Disclosure Opinion",
        "snippet": "Opinion on securities disclosure obligations for IPO.",
        "doc_type": "opinion",
        "jurisdiction": "usa",
        "tags": ["securities", "sec", "ipo", "disclosure"],
        "content": "Legal opinion on Form S-1 disclosure requirements for proposed IPO. Reviews risk factors, MD&A, executive compensation disclosure, and compliance with Regulation S-K. Registration statement appears compliant.",
    },
    {
        "id": "US-OP-003",
        "title": "Tax Opinion - REIT Qualification",
        "snippet": "Opinion on real estate investment trust tax qualification.",
        "doc_type": "opinion",
        "jurisdiction": "usa",
        "tags": ["tax", "reit", "irs", "real-estate"],
        "content": "Tax opinion analyzing client's qualification as a REIT under IRC Sections 856-860. Reviews asset tests, income tests, distribution requirements, and shareholder requirements. Entity should qualify for REIT status.",
    },
]


def get_corpus() -> list[dict[str, Any]]:
    """Return the full corpus."""
    return CORPUS


def get_by_jurisdiction(jurisdiction: str) -> list[dict[str, Any]]:
    """Filter corpus by jurisdiction ('brazil' or 'usa')."""
    return [doc for doc in CORPUS if doc["jurisdiction"] == jurisdiction]


def get_by_doc_type(doc_type: str) -> list[dict[str, Any]]:
    """Filter corpus by document type ('contract', 'brief', 'opinion')."""
    return [doc for doc in CORPUS if doc["doc_type"] == doc_type]
