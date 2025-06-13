# Dashboard Feature - User Guide

## Introduction

The Dashboard feature in the Academic AI Assistant provides powerful analytics and management tools to help you track student progress, monitor system usage, and manage user data. This guide will walk you through using all aspects of the Dashboard.

## Accessing the Dashboard

1. Launch the Academic AI Assistant application
2. In the sidebar navigation menu, click on "📊 Dashboard"
3. The Dashboard will load with the "System Overview" tab displayed by default

## Navigating the Dashboard

The Dashboard is organized into four main tabs:

- **System Overview**: High-level metrics and trends
- **Student Leaderboard**: Ranking of students by performance
- **Activity Metrics**: Detailed activity analysis
- **Performance Analytics**: Academic performance trends

You can switch between tabs by clicking on the tab names at the top of the Dashboard.

## System Overview Tab

This tab provides a quick snapshot of system-wide metrics and trends.

### Key Features:

1. **Summary Metrics**
   - View total counts of students, notes, tasks, and quizzes
   - These metrics update automatically as new data is added to the system

2. **System Activity Chart**
   - The line chart shows notes and tasks creation over the past 30 days
   - Hover over data points to see exact values for specific dates

3. **Subject Distribution**
   - The bar chart shows which subjects have the most notes
   - This helps identify popular or well-documented subjects

## Student Leaderboard Tab

This tab ranks students based on their activity and performance metrics.

### Key Features:

1. **Leaderboard Table**
   - View all students ranked by their overall score
   - The table includes columns for:
     - Name and learning style
     - Notes count
     - Tasks (total and completed)
     - Quizzes taken and average score
     - Productivity and overall scores
   - Click on any column header to sort the table by that metric

2. **Top Performers Section**
   - Quickly identify the top three students
   - See their key metrics at a glance

## Activity Metrics Tab

This tab provides detailed activity analysis for individual students or the entire system.

### Key Features:

1. **Student Selection**
   - Use the dropdown menu to select a specific student or "All Students"
   - The charts and metrics will update automatically based on your selection

2. **Notes Creation Activity**
   - The line chart shows note creation over time
   - Identify patterns in study habits and productivity

3. **Task Completion Metrics**
   - The pie chart shows the distribution of task statuses
   - Quickly see the ratio of completed vs. pending tasks

## Performance Analytics Tab

This tab focuses on academic performance metrics, particularly quiz results.

### Key Features:

1. **Student Selection for Quiz Analysis**
   - Use the dropdown to select a specific student or view all students
   - The performance charts will update based on your selection

2. **Quiz Performance Trends**
   - The scatter plot shows quiz scores over time
   - The trend line indicates whether performance is improving or declining
   - The system will tell you if performance is improving or declining, with the trend slope
   - The table below shows detailed quiz results with dates and scores

3. **Subject Proficiency**
   - The bar chart shows average quiz scores by subject
   - Identify strengths and weaknesses across different academic areas
   - The table shows the number of quizzes taken in each subject

## User Management Section

Below the main dashboard tabs is a User Management section for administrative functions.

### Deleting a Student Profile

**Warning**: This action permanently removes a student and all their associated data.

To delete a student:

1. Click on the "Delete Student Profile" expander
2. Select the student from the dropdown menu
3. Read the warning message carefully
4. Check the confirmation checkbox to confirm deletion
5. Click the "Delete Student" button
6. The system will display what data will be deleted
7. After deletion, click "Refresh Dashboard" to update the page

## Tips for Effective Dashboard Use

1. **Regular Monitoring**: Check the Dashboard periodically to track progress and identify trends
2. **Comparative Analysis**: Use the Student Leaderboard to identify high performers and their habits
3. **Subject Focus**: Use the Subject Proficiency chart to identify areas that need more attention
4. **Performance Tracking**: Monitor the Quiz Performance Trends to ensure continuous improvement
5. **Data Management**: Use the User Management section to maintain clean and accurate data

## Troubleshooting

1. **Missing Data**: If charts or tables appear empty, it may be because:
   - There is no data available yet
   - The selected student has no activity in that category
   - The database tables are not properly set up

2. **Error Messages**: If you see error messages:
   - Check that all required database tables exist
   - Ensure you have the necessary permissions
   - Try refreshing the page

3. **Visualization Issues**: If charts don't render properly:
   - Try selecting different date ranges or filters
   - Ensure you have a stable internet connection
   - Try using a different browser

## Getting Help

If you encounter issues with the Dashboard feature:

1. Check the documentation for known limitations
2. Look for error messages that may provide clues
3. Contact system administration for database-related issues
4. Submit a support ticket for persistent problems 