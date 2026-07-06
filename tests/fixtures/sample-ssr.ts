import {
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  TextRun,
} from "docx";

/**
 * A synthetic PharmD Self-Study Report (SSR) whose sections map to the
 * NCAAA program standards (Mission & Goals, Program Management & QA,
 * Teaching & Learning, Students, Teaching Staff, Learning Resources).
 * Deliberately uneven: some standards are addressed well, others weakly
 * or not at all, so the AI Reviewer produces a spread of verdicts.
 */
export async function buildSampleSSR(): Promise<Buffer> {
  const h1 = (text: string) =>
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
  const p = (text: string) => new Paragraph({ children: [new TextRun(text)] });

  const doc = new Document({
    sections: [
      {
        children: [
          new Paragraph({
            heading: HeadingLevel.TITLE,
            children: [new TextRun("Self-Study Report — Doctor of Pharmacy (PharmD) Program")],
          }),
          p("Prepared by the College of Pharmacy, Demo University, for NCAAA program accreditation."),

          h1("1. Mission and Goals"),
          p(
            "The PharmD program mission is to graduate competent, ethical pharmacists prepared for patient-centered care and lifelong learning. The mission was approved by the College Council in 2021 and is aligned with the university mission and the National Transformation goals for the health sector."
          ),
          p(
            "Program goals are documented and measurable: (G1) deliver an accredited curriculum meeting professional competency standards; (G2) achieve graduate employability above 85% within one year; (G3) expand experiential training partnerships. Each goal is reviewed annually by the Program Quality Committee and evidence of achievement is retained."
          ),
          p(
            "Stakeholders — faculty, students, employers, and the professional advisory board — were consulted through documented surveys and minutes when the mission and goals were last revised."
          ),

          h1("2. Program Management and Quality Assurance"),
          p(
            "The program is governed by a Program Committee chaired by the Program Coordinator. A quality management plan defines the PDCA cycle, key performance indicators, and an annual program report. KPIs are benchmarked against internal targets and peer programs, and results feed documented improvement actions."
          ),
          p(
            "An annual program report is produced and reviewed; the most recent report identified improvement actions with owners and deadlines. Risk management, however, is described only in general terms and lacks a formal risk register."
          ),

          h1("3. Teaching and Learning"),
          p(
            "Program learning outcomes (PLOs) are defined across knowledge, skills, and values domains and mapped to course learning outcomes (CLOs) in a curriculum map. Assessment methods are aligned to outcomes and moderated. Direct and indirect measures of learning outcome achievement are collected each semester and analyzed."
          ),
          p(
            "Course specifications and reports follow the approved templates. Teaching strategies include problem-based learning, laboratory practice, and clinical rotations. Benchmarking of student achievement against comparable programs is performed for selected courses."
          ),

          h1("4. Students"),
          p(
            "Admission criteria are published and applied consistently. Academic advising is provided, and student progression and completion rates are tracked. Student satisfaction is surveyed annually."
          ),

          h1("5. Teaching Staff"),
          p(
            "The program employs qualified academic staff with appropriate ratios. Professional development is available but participation is not systematically recorded, and the workload policy is not consistently documented across departments."
          ),

          // Standard 6 (Learning Resources, Facilities, Equipment) is
          // intentionally omitted so the reviewer marks it Not Met.
        ],
      },
    ],
  });

  return Buffer.from(await Packer.toBuffer(doc));
}
