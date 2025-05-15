export type Summary = string;

export interface SummaryResponse {
    summary: Summary;
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
