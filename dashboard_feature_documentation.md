# Dashboard Feature Documentation - Academic AI Assistant

## Overview

The Dashboard feature provides a comprehensive analytics platform for monitoring student progress, activity metrics, and performance trends across the Academic AI Assistant system. It serves as a central hub for visualizing educational data and deriving actionable insights.

## Key Components

The Dashboard is organized into four main tabs, each focusing on different aspects of student performance and system usage:

### 1. System Overview

This tab provides a high-level summary of system-wide metrics and trends:

- **Summary Metrics**: Displays total counts of students, notes, tasks, and quizzes in the system
- **System Activity Over Time**: Line chart showing notes and tasks creation activity over the past 30 days
- **Subject Distribution**: Bar chart showing the distribution of notes across different academic subjects

### 2. Student Leaderboard

This tab ranks students based on their activity and performance:

- **Leaderboard Table**: Sortable table with the following metrics for each student:
  - Name and learning style
  - Notes count
  - Tasks (total and completed)
  - Quizzes taken and average score
  - Productivity score (calculated as notes × 5 + completed tasks × 3)
  - Overall score (combining productivity and quiz performance)
- **Top Performers**: Highlights the top three students with their key metrics

### 3. Activity Metrics

This tab provides detailed activity analysis for individual students or the entire system:

- **Student Selection**: Dropdown to filter data for a specific student or view all students
- **Notes Creation Activity**: Line chart showing note creation over time
- **Task Completion Metrics**: Pie chart showing the distribution of task statuses (pending vs. completed)

### 4. Performance Analytics

This tab focuses on academic performance metrics:

- **Quiz Performance Trends**: 
  - Scatter plot with trend line showing quiz scores over time
  - Trend analysis indicating whether performance is improving or declining
  - Detailed table of quiz results with dates and scores
- **Subject Proficiency**: 
  - Table and bar chart showing average quiz scores by subject
  - Helps identify strengths and weaknesses across different academic areas

## User Management

Below the main dashboard tabs is a User Management section with administrative functions:

- **Delete Student Profile**: Allows removing a student and all associated data
  - Includes confirmation steps to prevent accidental deletion
  - Provides a summary of data to be deleted (notes, tasks, quizzes)
  - Handles the deletion process with appropriate database operations

## Technical Implementation

The Dashboard feature leverages several technologies:

- **Data Visualization**: Uses Matplotlib and Streamlit's native charting capabilities
- **Database Queries**: Employs PostgreSQL queries to aggregate and analyze student data
- **Interactive Components**: Implements Streamlit widgets for filtering and user interaction
- **Performance Calculations**: Includes algorithms for calculating productivity scores and trend analysis

## Database Integration

The Dashboard connects to the following database tables:

- `students`: For student profile information
- `notes`: For tracking note creation and subjects
- `tasks`: For monitoring task completion rates
- `quizzes`: For analyzing quiz performance and subject proficiency
- `knowledge_base`: For additional student-related data

## Usage Scenarios

### For Instructors/Administrators:
- Monitor overall system usage and engagement
- Identify high-performing and struggling students
- Track subject areas that need additional attention
- Manage student profiles and data

### For Students:
- View personal performance metrics and trends
- Compare performance against peers (leaderboard)
- Identify personal strengths and areas for improvement

## Future Enhancements

Potential improvements for the Dashboard feature:

1. **Advanced Analytics**: Implement predictive analytics to forecast student performance
2. **Custom Date Ranges**: Allow filtering data by custom time periods
3. **Export Capabilities**: Add options to export reports as PDF or CSV
4. **Notification System**: Alert instructors about significant trends or issues
5. **Personalized Recommendations**: Generate AI-powered study recommendations based on performance data

## Technical Notes

- The dashboard handles type conversion for database values (e.g., converting Decimal to float)
- Error handling is implemented for missing tables or columns
- The feature includes data visualization best practices for clear interpretation
- User management operations are designed with appropriate safeguards 