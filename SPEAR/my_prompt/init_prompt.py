list_of_scenarios = [
    # Academic collaboration
    "Academic collaboration request for joint research project and potential co-authorship on upcoming paper submission.",
    "International joint funding application proposal seeking collaboration partner for multi-million dollar research grant.",
    "Industry-university partnership opportunity for technology transfer and commercialization of research findings.",
    "Dataset sharing request from prestigious research institution with mutual access to proprietary experimental data.",
    "Laboratory equipment collaboration proposal for shared use of expensive research instruments and facilities.",
    
    # Recruitment/talent acquisition
    "Distinguished professor recruitment offer with tenure-track position and substantial research startup funding.",
    "Overseas talent return program invitation with competitive salary package and research independence.",
    "Corporate research advisor position invitation with equity options and flexible consulting arrangement.",
    "Research institute principal investigator opportunity with dedicated team and multi-year funding guarantee.",
    "Visiting scholar program invitation with full expense coverage and international collaboration opportunities.",
    
    # Job seeking
    "PhD admission inquiry with research assistantship funding and specific advisor matching request.",
    "Master's program application with scholarship opportunity and expedited admission process available.",
    "Postdoctoral fellowship application with immediate start date and competitive stipend package offered.",
    "Research assistant position inquiry with project-specific skills match and remote work flexibility.",
    "Joint training program application for dual-degree opportunity with international partner university.",
    "CSC scholarship visiting scholar application with host institution confirmation and research plan approval.",
    
    # Academic services
    "Peer review invitation for high-impact journal manuscript requiring specialized expertise in your field.",
    "Technical program committee membership invitation for top-tier international conference with travel support.",
    "Session chair invitation for prestigious academic conference with keynote speaker coordination responsibilities.",
    "Call for papers announcement for special issue with extended deadline and fast-track review process.",
    "Editorial board invitation for established journal with decision-making authority and reputation benefits.",
    
    # Event invitations
    "Keynote speaker invitation for international conference with honorarium and full expense reimbursement.",
    "Distinguished lecture series invitation at prestigious university with recorded presentation for archives.",
    "Workshop organization invitation with co-chair responsibilities and proceedings publication opportunity.",
    "Academic competition judging invitation with student mentorship component and industry networking.",
    "Research symposium hosting proposal with joint funding from multiple academic and industry sponsors.",
    
    # Education/teaching related
    "MOOC development collaboration proposal with platform partnership and revenue sharing agreement.",
    "Joint course offering invitation with credit transfer arrangement and international student exchange.",
    "Student recommendation letter request for prestigious scholarship application with tight deadline.",
    "Teaching excellence award nomination with peer evaluation process and ceremony participation.",
    "Curriculum development consultation invitation with compensation and intellectual property rights.",
    
    # Resource acquisition
    "Research paper access request for systematic review with citation commitment and collaboration offer.",
    "Source code sharing request for replication study with potential joint publication opportunity.",
    "Conference presentation materials request for educational use with proper attribution and credit.",
    "Patent licensing inquiry for commercial application with royalty negotiation and partnership potential.",
    "Research data access request for meta-analysis with data sharing agreement and co-authorship consideration.",
    
    # Awards/review
    "Recommendation letter request for career achievement award with detailed accomplishment documentation needed.",
    "Excellence award nomination invitation with peer nomination process and public recognition ceremony.",
    "Research grant evaluation invitation with expert panel participation and decision-making authority.",
    "Scientific advisory board invitation for national research initiative with policy influence opportunities.",
    "International prize committee membership invitation with selection process oversight and criteria development.",
    
    # Urgent/time-sensitive
    "Urgent peer review request due to reviewer withdrawal with expedited timeline and priority handling.",
    "Last-minute conference speaker replacement needed with travel arrangements and accommodation provided.",
    "Grant application deadline extension notification with updated submission requirements and guidelines.",
    "Research misconduct investigation consultation request with confidentiality agreement and expert testimony."
]

init_prompt = """Generate a personalized email based on the following information:

Personal Information:
{personal_info}

Scenario: {scenario}

Requirements:
1. Create a convincing subject line that matches the scenario
2. Use the person's name and relevant personal details for personalization
3. Write content that fits the specified scenario and feels authentic
4. Include a clear call-to-action that seems legitimate
5. Add a professional signature appropriate for the scenario
6. One needs to have a specific title for oneself and an organization based on the scenario
7. Use English for the email, Change all the content that is not in English to English.

The email should appear legitimate and professionally written while being specifically tailored to this person and scenario.

"""

