import re
from typing import Annotated, List, Optional

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, ValidationInfo, field_validator

_MONTHS = {name: number for number, name in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def _in_source(value: str, info: ValidationInfo) -> str:
    """Reject values absent from the source text, e.g. names leaked from the few-shot examples.

    Instructor sends the error back to the model. The check is skipped without a validation context.
    """
    source = (info.context or {}).get("resume_text")
    if source is not None and _squash(value) not in _squash(source):
        raise ValueError(f"{value!r} is not in the resume text. Copy it verbatim or leave it out.")
    return value


def _normalize_date(value):
    """Turn 'Jun 2021' into '2021-06', 'current' into 'Present', and the text 'null' into None.

    Schema-constrained models can emit null as a string. Anything else stays unchanged.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.casefold() in {"", "null", "none", "n/a"}:
        return None
    if text.casefold() in {"present", "current", "now"}:
        return "Present"
    match = re.fullmatch(r"([A-Za-z]{3,9})\.?,?\s+(\d{4})", text)
    if match and (month := _MONTHS.get(match[1][:3].lower())):
        return f"{match[2]}-{month:02d}"
    return text


Verbatim = Annotated[str, AfterValidator(_in_source)]
DateStr = Annotated[Optional[str], BeforeValidator(_normalize_date)]


class ContactInfo(BaseModel):
    name: Optional[Verbatim] = None
    email: Optional[Verbatim] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class Experience(BaseModel):
    company: Verbatim
    title: str
    start_date: DateStr = Field(None, description="YYYY-MM if known")
    end_date: DateStr = Field(None, description="YYYY-MM or 'Present'")
    location: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: Verbatim
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: DateStr = None
    end_date: DateStr = None
    gpa: Optional[str] = None


class Project(BaseModel):
    name: Verbatim
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    link: Optional[str] = None


class Certification(BaseModel):
    name: Verbatim
    issuer: Optional[str] = None
    date: DateStr = None


class Resume(BaseModel):
    contact: ContactInfo
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)

    @field_validator("skills")
    @classmethod
    def dedupe_skills(cls, skills: List[str]) -> List[str]:
        unique = {}
        for skill in skills:
            unique.setdefault(skill.strip().casefold(), skill.strip())
        return list(unique.values())


class Answer(BaseModel):
    """Answer to a question about a resume, grounded in quotes from the retrieved text.

    Field order matters: Ollama generates in schema order, and with answer first llama3.2 closed
    the string at once and returned an empty answer. Quotes first makes it find the evidence first.
    """

    quotes: List[Verbatim] = Field(  # required: llama3.2 skips this field when it is optional
        description="Exact passages supporting the answer. Empty if the text does not answer the question.",
    )
    answer: str = Field(min_length=1)
