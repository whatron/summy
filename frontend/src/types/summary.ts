export interface Summary {
    main_concepts: string[];
    key_definitions: string[];
    important_formulas: string[];
    examples: string[];
    learning_objectives: string[];
    prerequisites: string[];
    difficulty_level: string;
    confidence_score: number;
    sentiment: string;
}

export interface LectureMetadata {
    subject: string;
    topic: string;
    level: string;
    estimated_duration: string;
}

export interface SummaryResponse {
    summary: Summary;
    lecture_metadata: LectureMetadata;
    transcript_id: string;
    processing_time: number;
    error?: string;
}

export interface ProcessRequest {
    transcript: string;
    metadata?: {
        meeting_title?: string;
        date?: string;
        duration?: number;
        participants?: string[];
        language?: string;
        context?: {
            industry?: string;
            project_name?: string;
            meeting_type?: 'one_on_one' | 'team_meeting' | 'client_meeting' | 'other';
            priority?: 'high' | 'medium' | 'low';
        };
    };
    options?: {
        include_sentiment?: boolean;
        include_confidence_score?: boolean;
        max_summary_length?: number;
        format?: 'bullet_points' | 'paragraph' | 'both';
    };
}

export interface TranscriptSegment {
    text: string;
    start_time: number;
    end_time: number;
    speaker?: string;
    confidence?: number;
}

export interface TranscriptResponse {
    segments: TranscriptSegment[];
    full_text: string;
    duration: number;
    language?: string;
    speakers?: string[];
}
