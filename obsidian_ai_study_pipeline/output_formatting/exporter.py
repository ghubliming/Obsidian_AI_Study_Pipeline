"""
Output formatting module for exporting quiz questions to various formats.
"""

import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

from ..generation import QuizQuestion, QuizType

logger = logging.getLogger(__name__)

class QuizExporter:
    """Export quiz questions to various formats."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize quiz exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized QuizExporter with output directory: {output_dir}")
    
    def export_to_markdown(self, 
                          questions: List[QuizQuestion],
                          filename: Optional[str] = None,
                          group_by_source: bool = True) -> str:
        """
        Export quiz questions to Markdown format.
        
        Args:
            questions: List of quiz questions to export
            filename: Output filename (auto-generated if None)
            group_by_source: Whether to group questions by source note
            
        Returns:
            Path to the exported file
        """
        if not questions:
            logger.warning("No questions to export to Markdown")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quiz_questions_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# Quiz Questions\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total questions: {len(questions)}\n\n")
            
            # Write table of contents
            f.write("## Table of Contents\n\n")
            if group_by_source:
                sources = list(set(q.source_note for q in questions))
                for i, source in enumerate(sorted(sources), 1):
                    f.write(f"{i}. [{source}](#{source.lower().replace(' ', '-').replace('_', '-')})\n")
            else:
                f.write("1. [All Questions](#all-questions)\n")
            f.write("\n---\n\n")
            
            if group_by_source:
                self._write_grouped_questions(f, questions)
            else:
                self._write_all_questions(f, questions)
        
        logger.info(f"Exported {len(questions)} questions to Markdown: {filepath}")
        return filepath
    
    def _write_grouped_questions(self, f, questions: List[QuizQuestion]):
        """Write questions grouped by source note."""
        # Group questions by source
        grouped = {}
        for q in questions:
            if q.source_note not in grouped:
                grouped[q.source_note] = []
            grouped[q.source_note].append(q)
        
        for source_note in sorted(grouped.keys()):
            source_questions = grouped[source_note]
            
            f.write(f"## {source_note}\n\n")
            f.write(f"**Source Path:** `{source_questions[0].source_path}`\n")
            f.write(f"**Number of Questions:** {len(source_questions)}\n\n")
            
            for i, question in enumerate(source_questions, 1):
                self._write_single_question_md(f, question, i)
            
            f.write("\n---\n\n")
    
    def _write_all_questions(self, f, questions: List[QuizQuestion]):
        """Write all questions in a single section."""
        f.write("## All Questions\n\n")
        
        for i, question in enumerate(questions, 1):
            self._write_single_question_md(f, question, i)
    
    def _write_single_question_md(self, f, question: QuizQuestion, index: int):
        """Write a single question in Markdown format."""
        f.write(f"### Question {index}\n\n")
        f.write(f"**Type:** {question.quiz_type.value.replace('_', ' ').title()}\n")
        f.write(f"**Source:** {question.source_note}\n\n")
        
        f.write(f"**Question:** {question.question}\n\n")
        
        if question.quiz_type == QuizType.MULTIPLE_CHOICE and question.options:
            f.write("**Options:**\n")
            for option in question.options:
                f.write(f"- {option}\n")
            f.write("\n")
        
        f.write(f"**Answer:** {question.answer}\n\n")
        
        if question.explanation:
            f.write(f"**Explanation:** {question.explanation}\n\n")
        
        if question.tags:
            f.write(f"**Tags:** {', '.join(question.tags)}\n\n")
        
        f.write(f"**Source Content:** {question.source_chunk}\n\n")
        f.write("---\n\n")
    
    def export_to_quizlet_csv(self, 
                             questions: List[QuizQuestion],
                             filename: Optional[str] = None) -> str:
        """
        Export quiz questions to Quizlet-compatible CSV format.
        
        Args:
            questions: List of quiz questions to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the exported file
        """
        if not questions:
            logger.warning("No questions to export to Quizlet CSV")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quizlet_flashcards_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Quizlet CSV format: Term, Definition
            writer = csv.writer(f)
            writer.writerow(['Term', 'Definition'])
            
            for question in questions:
                term, definition = self._convert_to_flashcard_format(question)
                writer.writerow([term, definition])
        
        logger.info(f"Exported {len(questions)} questions to Quizlet CSV: {filepath}")
        return filepath
    
    def _convert_to_flashcard_format(self, question: QuizQuestion) -> tuple:
        """Convert a quiz question to flashcard term/definition format."""
        if question.quiz_type == QuizType.FLASHCARD:
            return question.question, question.answer
        
        elif question.quiz_type == QuizType.MULTIPLE_CHOICE:
            # For MC, use question as term and answer with explanation as definition
            term = question.question
            if question.options:
                term += "\n\nOptions:\n" + "\n".join(question.options)
            
            definition = f"Answer: {question.answer}"
            if question.explanation:
                definition += f"\n\nExplanation: {question.explanation}"
            
            return term, definition
        
        elif question.quiz_type == QuizType.CLOZE_DELETION:
            # Use the question with blank as term, answer as definition
            return question.question, question.answer
        
        elif question.quiz_type == QuizType.SHORT_ANSWER:
            return question.question, question.answer
        
        elif question.quiz_type == QuizType.TRUE_FALSE:
            term = f"{question.question} (True/False)"
            definition = f"{question.answer}"
            if question.explanation:
                definition += f"\n\nExplanation: {question.explanation}"
            return term, definition
        
        else:
            return question.question, question.answer
    
    def export_to_anki_csv(self, 
                          questions: List[QuizQuestion],
                          filename: Optional[str] = None) -> str:
        """
        Export quiz questions to Anki-compatible CSV format.
        
        Args:
            questions: List of quiz questions to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the exported file
        """
        if not questions:
            logger.warning("No questions to export to Anki CSV")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anki_cards_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Anki CSV format: Front, Back, Tags
            writer = csv.writer(f)
            
            for question in questions:
                front, back = self._convert_to_flashcard_format(question)
                tags = ' '.join(question.tags) if question.tags else ''
                
                # Add source information to tags
                source_tag = f"source::{question.source_note.replace(' ', '_')}"
                type_tag = f"type::{question.quiz_type.value}"
                
                all_tags = f"{tags} {source_tag} {type_tag}".strip()
                
                writer.writerow([front, back, all_tags])
        
        logger.info(f"Exported {len(questions)} questions to Anki CSV: {filepath}")
        return filepath
    
    def export_to_json(self, 
                      questions: List[QuizQuestion],
                      filename: Optional[str] = None,
                      include_metadata: bool = True) -> str:
        """
        Export quiz questions to JSON format.
        
        Args:
            questions: List of quiz questions to export
            filename: Output filename (auto-generated if None)
            include_metadata: Whether to include full metadata
            
        Returns:
            Path to the exported file
        """
        if not questions:
            logger.warning("No questions to export to JSON")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quiz_questions_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert questions to dictionaries
        questions_data = []
        for q in questions:
            q_dict = {
                'question': q.question,
                'answer': q.answer,
                'quiz_type': q.quiz_type.value,
                'source_note': q.source_note,
                'source_path': q.source_path,
                'options': q.options,
                'explanation': q.explanation,
                'difficulty': q.difficulty,
                'tags': q.tags,
                'confidence_score': q.confidence_score
            }
            
            if include_metadata:
                q_dict['source_chunk'] = q.source_chunk
            
            questions_data.append(q_dict)
        
        # Create export data with metadata
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_questions': len(questions),
                'question_types': self._get_type_counts(questions),
                'source_notes': list(set(q.source_note for q in questions))
            },
            'questions': questions_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(questions)} questions to JSON: {filepath}")
        return filepath
    
    def export_study_guide(self, 
                          questions: List[QuizQuestion],
                          filename: Optional[str] = None,
                          include_answers: bool = False) -> str:
        """
        Export questions as a study guide (questions only, optionally with answers).
        
        Args:
            questions: List of quiz questions to export
            filename: Output filename (auto-generated if None)
            include_answers: Whether to include answers in the study guide
            
        Returns:
            Path to the exported file
        """
        if not questions:
            logger.warning("No questions to export as study guide")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_with_answers" if include_answers else "_questions_only"
            filename = f"study_guide{suffix}_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Study Guide\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total questions: {len(questions)}\n\n")
            
            if not include_answers:
                f.write("*Note: This study guide contains questions only. "
                       "Refer to the full quiz export for answers.*\n\n")
            
            f.write("---\n\n")
            
            # Group by source for better organization
            grouped = {}
            for q in questions:
                if q.source_note not in grouped:
                    grouped[q.source_note] = []
                grouped[q.source_note].append(q)
            
            for source_note in sorted(grouped.keys()):
                source_questions = grouped[source_note]
                
                f.write(f"## {source_note}\n\n")
                
                for i, question in enumerate(source_questions, 1):
                    f.write(f"**{i}.** {question.question}\n\n")
                    
                    if question.quiz_type == QuizType.MULTIPLE_CHOICE and question.options:
                        for option in question.options:
                            f.write(f"   {option}\n")
                        f.write("\n")
                    
                    if include_answers:
                        f.write(f"   *Answer: {question.answer}*\n")
                        if question.explanation:
                            f.write(f"   *Explanation: {question.explanation}*\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
        
        guide_type = "with answers" if include_answers else "questions only"
        logger.info(f"Exported study guide ({guide_type}) with {len(questions)} questions: {filepath}")
        return filepath
    
    def _get_type_counts(self, questions: List[QuizQuestion]) -> Dict[str, int]:
        """Get count of each question type."""
        counts = {}
        for q in questions:
            counts[q.quiz_type.value] = counts.get(q.quiz_type.value, 0) + 1
        return counts
    
    def create_export_summary(self, questions: List[QuizQuestion]) -> Dict:
        """Create a summary of questions for export metadata."""
        if not questions:
            return {}
        
        sources = set(q.source_note for q in questions)
        types = self._get_type_counts(questions)
        
        difficulty_counts = {}
        confidence_scores = []
        
        for q in questions:
            difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
            confidence_scores.append(q.confidence_score)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_questions': len(questions),
            'unique_sources': len(sources),
            'source_notes': list(sources),
            'question_types': types,
            'difficulty_distribution': difficulty_counts,
            'average_confidence': avg_confidence,
            'export_timestamp': datetime.now().isoformat()
        }