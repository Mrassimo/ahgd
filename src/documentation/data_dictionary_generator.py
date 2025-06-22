#!/usr/bin/env python3
"""Data Dictionary Generator for AHGD schemas.

This module provides automated documentation generation from Pydantic v2 schemas,
creating comprehensive data dictionaries for Australian health and geographic data.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import os
import json
import csv
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, get_origin, get_args
from datetime import datetime
from collections import defaultdict
import re

try:
    from pydantic import BaseModel, Field
    from pydantic.fields import FieldInfo
    from pydantic._internal._model_construction import complete_model_class
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = None
    FieldInfo = object

from ..utils.logging import get_logger


class SchemaFieldInfo:
    """Information about a schema field for documentation."""
    
    def __init__(
        self,
        field_name: str,
        field_type: str,
        description: str = "",
        constraints: Dict[str, Any] = None,
        examples: List[str] = None,
        source: str = "",
        validation_rules: List[str] = None,
        relationships: List[str] = None
    ):
        self.field_name = field_name
        self.field_type = field_type
        self.description = description
        self.constraints = constraints or {}
        self.examples = examples or []
        self.source = source
        self.validation_rules = validation_rules or []
        self.relationships = relationships or []


class SchemaInfo:
    """Information about a complete schema for documentation."""
    
    def __init__(
        self,
        schema_name: str,
        description: str = "",
        category: str = "",
        fields: List[SchemaFieldInfo] = None,
        relationships: List[str] = None,
        source_file: str = ""
    ):
        self.schema_name = schema_name
        self.description = description
        self.category = category
        self.fields = fields or []
        self.relationships = relationships or []
        self.source_file = source_file


class DataDictionaryGenerator:
    """Automated data dictionary generator for AHGD Pydantic schemas."""
    
    def __init__(self, schemas_dir: Union[str, Path] = "schemas"):
        """Initialise the data dictionary generator.
        
        Args:
            schemas_dir: Directory containing Pydantic schema files
        """
        self.schemas_dir = Path(schemas_dir)
        self.logger = get_logger(__name__)
        self.schemas_info: List[SchemaInfo] = []
        
        # Australian health data context
        self.australian_context = {
            'sa2_code': {
                'description': 'Statistical Area Level 2 code (ASGS 2021)',
                'source': 'Australian Bureau of Statistics',
                'format': '11-digit code (SSSAASSSSS)',
                'example': '101011007'
            },
            'seifa_scores': {
                'description': 'Socio-Economic Indexes for Areas',
                'source': 'Australian Bureau of Statistics',
                'range': 'Typically 600-1200',
                'methodology': 'Principal component analysis of Census variables'
            },
            'life_expectancy': {
                'description': 'Life expectancy at birth',
                'source': 'Australian Institute of Health and Welfare',
                'range': 'Typically 70-90 years',
                'unit': 'Years'
            },
            'remoteness_category': {
                'description': 'ARIA (Accessibility/Remoteness Index of Australia)',
                'source': 'Department of Health',
                'categories': ['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote']
            }
        }
        
        if not PYDANTIC_AVAILABLE:
            self.logger.warning("Pydantic not available - limited functionality")
    
    def discover_schemas(self) -> List[Type[BaseModel]]:
        """Discover all Pydantic schema classes in the schemas directory.
        
        Returns:
            List of discovered Pydantic model classes
        """
        if not PYDANTIC_AVAILABLE:
            self.logger.error("Pydantic not available for schema discovery")
            return []
        
        schema_classes = []
        
        if not self.schemas_dir.exists():
            self.logger.warning(f"Schemas directory not found: {self.schemas_dir}")
            return []
        
        # Import all Python files in schemas directory
        for schema_file in self.schemas_dir.rglob("*.py"):
            if schema_file.name.startswith("__"):
                continue
                
            try:
                # Get relative path for module import
                rel_path = schema_file.relative_to(self.schemas_dir.parent)
                module_path = str(rel_path.with_suffix("")).replace("/", ".")
                
                # Import the module
                module = __import__(module_path, fromlist=[""])
                
                # Find Pydantic models in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseModel) and 
                        obj != BaseModel):
                        schema_classes.append(obj)
                        self.logger.info(f"Discovered schema: {name}")
                        
            except Exception as e:
                self.logger.warning(f"Could not import {schema_file}: {str(e)}")
        
        return schema_classes
    
    def extract_field_info(self, model_class: Type[BaseModel], field_name: str, field_info: FieldInfo) -> SchemaFieldInfo:
        """Extract comprehensive information about a model field.
        
        Args:
            model_class: The Pydantic model class
            field_name: Name of the field
            field_info: Pydantic FieldInfo object
            
        Returns:
            SchemaFieldInfo object with extracted information
        """
        # Get field type
        field_type = self._format_field_type(field_info.annotation)
        
        # Get description from field info or docstring
        description = field_info.description or ""
        
        # Add Australian context if available
        if field_name.lower() in self.australian_context:
            context = self.australian_context[field_name.lower()]
            if description:
                description += f" ({context['description']})"
            else:
                description = context['description']
        
        # Extract constraints
        constraints = {}
        if hasattr(field_info, 'constraints'):
            for constraint_name, constraint_value in field_info.constraints.items():
                if constraint_value is not None:
                    constraints[constraint_name] = constraint_value
        
        # Generate examples
        examples = self._generate_examples(field_name, field_type, constraints)
        
        # Get data source
        source = self._get_field_source(field_name)
        
        # Extract validation rules
        validation_rules = self._extract_validation_rules(field_info)
        
        return SchemaFieldInfo(
            field_name=field_name,
            field_type=field_type,
            description=description,
            constraints=constraints,
            examples=examples,
            source=source,
            validation_rules=validation_rules
        )
    
    def _format_field_type(self, annotation: Any) -> str:
        """Format field type annotation as readable string."""
        if annotation is None:
            return "Any"
        
        # Handle Union types (including Optional)
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            if len(args) == 2 and type(None) in args:
                # Optional type
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return f"Optional[{self._format_field_type(non_none_type)}]"
            else:
                # Union type
                type_names = [self._format_field_type(arg) for arg in args]
                return f"Union[{', '.join(type_names)}]"
        
        # Handle List types
        if origin is list:
            args = get_args(annotation)
            if args:
                return f"List[{self._format_field_type(args[0])}]"
            return "List"
        
        # Handle Dict types
        if origin is dict:
            args = get_args(annotation)
            if len(args) >= 2:
                return f"Dict[{self._format_field_type(args[0])}, {self._format_field_type(args[1])}]"
            return "Dict"
        
        # Handle basic types
        if hasattr(annotation, '__name__'):
            return annotation.__name__
        
        return str(annotation)
    
    def _generate_examples(self, field_name: str, field_type: str, constraints: Dict[str, Any]) -> List[str]:
        """Generate realistic examples for Australian health data fields."""
        examples = []
        
        field_name_lower = field_name.lower()
        
        # Australian-specific examples
        if 'sa2_code' in field_name_lower:
            examples = ['101011007', '201011021', '301011001']
        elif 'postcode' in field_name_lower:
            examples = ['3000', '2000', '4000', '5000', '6000']
        elif 'state' in field_name_lower:
            examples = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
        elif 'life_expectancy' in field_name_lower:
            examples = ['82.1', '79.8', '84.3']
        elif 'seifa' in field_name_lower:
            examples = ['1156', '987', '1034']
        elif 'population' in field_name_lower:
            examples = ['5432', '12876', '3241']
        elif field_type == 'float':
            if 'rate' in field_name_lower or 'percentage' in field_name_lower:
                examples = ['15.2', '8.7', '23.1']
            else:
                examples = ['42.5', '67.8', '91.2']
        elif field_type == 'int':
            examples = ['123', '456', '789']
        elif field_type == 'str':
            examples = ['Example text', 'Sample value', 'Data entry']
        elif field_type == 'bool':
            examples = ['true', 'false']
        
        return examples[:3]  # Limit to 3 examples
    
    def _get_field_source(self, field_name: str) -> str:
        """Get the data source for a field based on naming conventions."""
        field_name_lower = field_name.lower()
        
        if any(term in field_name_lower for term in ['sa2', 'sa3', 'sa4', 'postcode', 'census']):
            return 'Australian Bureau of Statistics (ABS)'
        elif any(term in field_name_lower for term in ['health', 'mortality', 'life_expectancy']):
            return 'Australian Institute of Health and Welfare (AIHW)'
        elif 'seifa' in field_name_lower:
            return 'Australian Bureau of Statistics (ABS)'
        elif any(term in field_name_lower for term in ['climate', 'weather', 'rainfall']):
            return 'Bureau of Meteorology (BOM)'
        else:
            return 'Multiple sources'
    
    def _extract_validation_rules(self, field_info: FieldInfo) -> List[str]:
        """Extract validation rules from field info."""
        rules = []
        
        if hasattr(field_info, 'constraints'):
            constraints = field_info.constraints
            
            if 'ge' in constraints:
                rules.append(f"Greater than or equal to {constraints['ge']}")
            if 'le' in constraints:
                rules.append(f"Less than or equal to {constraints['le']}")
            if 'min_length' in constraints:
                rules.append(f"Minimum length: {constraints['min_length']}")
            if 'max_length' in constraints:
                rules.append(f"Maximum length: {constraints['max_length']}")
            if 'regex' in constraints:
                rules.append(f"Must match pattern: {constraints['regex']}")
        
        return rules
    
    def analyse_schemas(self) -> None:
        """Analyse all discovered schemas and extract documentation information."""
        if not PYDANTIC_AVAILABLE:
            self.logger.error("Cannot analyse schemas - Pydantic not available")
            return
        
        schema_classes = self.discover_schemas()
        
        for schema_class in schema_classes:
            try:
                # Get schema information
                schema_name = schema_class.__name__
                description = schema_class.__doc__ or ""
                category = self._categorise_schema(schema_name)
                
                # Extract field information
                fields = []
                for field_name, field_info in schema_class.model_fields.items():
                    field_info_obj = self.extract_field_info(schema_class, field_name, field_info)
                    fields.append(field_info_obj)
                
                # Create schema info
                schema_info = SchemaInfo(
                    schema_name=schema_name,
                    description=description,
                    category=category,
                    fields=fields,
                    source_file=inspect.getfile(schema_class)
                )
                
                self.schemas_info.append(schema_info)
                self.logger.info(f"Analysed schema: {schema_name} ({len(fields)} fields)")
                
            except Exception as e:
                self.logger.error(f"Error analysing schema {schema_class.__name__}: {str(e)}")
    
    def _categorise_schema(self, schema_name: str) -> str:
        """Categorise schema based on naming conventions."""
        name_lower = schema_name.lower()
        
        if 'geographic' in name_lower or 'boundary' in name_lower:
            return 'Geographic'
        elif 'health' in name_lower or 'mortality' in name_lower:
            return 'Health'
        elif 'demographic' in name_lower or 'population' in name_lower:
            return 'Demographics'
        elif 'seifa' in name_lower:
            return 'Socio-Economic'
        elif 'master' in name_lower or 'integrated' in name_lower:
            return 'Integration'
        else:
            return 'Other'
    
    def generate_markdown(self, output_path: Union[str, Path]) -> None:
        """Generate markdown data dictionary."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# AHGD Data Dictionary\n\n")
            f.write("Comprehensive data dictionary for Australian Health Geography Data schemas.\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Table of contents
            f.write("## Table of Contents\n\n")
            categories = defaultdict(list)
            for schema in self.schemas_info:
                categories[schema.category].append(schema)
            
            for category, schemas in sorted(categories.items()):
                f.write(f"- [{category}](#{category.lower().replace(' ', '-')})\n")
                for schema in schemas:
                    f.write(f"  - [{schema.schema_name}](#{schema.schema_name.lower()})\n")
            f.write("\n")
            
            # Generate documentation for each category
            for category, schemas in sorted(categories.items()):
                f.write(f"## {category}\n\n")
                
                for schema in schemas:
                    f.write(f"### {schema.schema_name}\n\n")
                    if schema.description:
                        f.write(f"{schema.description}\n\n")
                    
                    f.write("| Field | Type | Description | Constraints | Examples | Source |\n")
                    f.write("|-------|------|-------------|-------------|----------|--------|\n")
                    
                    for field in schema.fields:
                        constraints_str = ", ".join([f"{k}: {v}" for k, v in field.constraints.items()])
                        examples_str = ", ".join(field.examples)
                        
                        f.write(f"| {field.field_name} | {field.field_type} | {field.description} | {constraints_str} | {examples_str} | {field.source} |\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Generated markdown data dictionary: {output_path}")
    
    def generate_html(self, output_path: Union[str, Path]) -> None:
        """Generate HTML data dictionary with styling."""
        output_path = Path(output_path)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AHGD Data Dictionary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .category {{
            background-color: white;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .category-header {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 8px 8px 0 0;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .schema {{
            padding: 20px;
            border-bottom: 1px solid #eee;
        }}
        .schema:last-child {{
            border-bottom: none;
        }}
        .schema-name {{
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .schema-description {{
            color: #666;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .field-name {{
            font-weight: bold;
            color: #e74c3c;
        }}
        .field-type {{
            color: #9b59b6;
            font-family: monospace;
        }}
        .examples {{
            font-family: monospace;
            color: #27ae60;
        }}
        .toc {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc h2 {{
            margin-top: 0;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 5px 0;
        }}
        .toc a {{
            text-decoration: none;
            color: #3498db;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AHGD Data Dictionary</h1>
        <p>Comprehensive data dictionary for Australian Health Geography Data schemas.</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
"""
        
        # Generate table of contents
        categories = defaultdict(list)
        for schema in self.schemas_info:
            categories[schema.category].append(schema)
        
        for category, schemas in sorted(categories.items()):
            html_content += f'            <li><a href="#{category.lower().replace(" ", "-")}">{category}</a>\n'
            html_content += '                <ul>\n'
            for schema in schemas:
                html_content += f'                    <li><a href="#{schema.schema_name.lower()}">{schema.schema_name}</a></li>\n'
            html_content += '                </ul>\n'
            html_content += '            </li>\n'
        
        html_content += """        </ul>
    </div>
"""
        
        # Generate content for each category
        for category, schemas in sorted(categories.items()):
            html_content += f"""    <div class="category" id="{category.lower().replace(' ', '-')}">
        <div class="category-header">{category}</div>
"""
            
            for schema in schemas:
                html_content += f"""        <div class="schema" id="{schema.schema_name.lower()}">
            <div class="schema-name">{schema.schema_name}</div>
"""
                if schema.description:
                    html_content += f'            <div class="schema-description">{schema.description}</div>\n'
                
                html_content += """            <table>
                <thead>
                    <tr>
                        <th>Field</th>
                        <th>Type</th>
                        <th>Description</th>
                        <th>Constraints</th>
                        <th>Examples</th>
                        <th>Source</th>
                    </tr>
                </thead>
                <tbody>
"""
                
                for field in schema.fields:
                    constraints_str = ", ".join([f"{k}: {v}" for k, v in field.constraints.items()])
                    examples_str = ", ".join(field.examples)
                    
                    html_content += f"""                    <tr>
                        <td class="field-name">{field.field_name}</td>
                        <td class="field-type">{field.field_type}</td>
                        <td>{field.description}</td>
                        <td>{constraints_str}</td>
                        <td class="examples">{examples_str}</td>
                        <td>{field.source}</td>
                    </tr>
"""
                
                html_content += """                </tbody>
            </table>
        </div>
"""
            
            html_content += "    </div>\n"
        
        html_content += """</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated HTML data dictionary: {output_path}")
    
    def generate_csv(self, output_path: Union[str, Path]) -> None:
        """Generate CSV data dictionary for spreadsheet use."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Schema', 'Category', 'Field', 'Type', 'Description',
                'Constraints', 'Examples', 'Source', 'Validation Rules'
            ])
            
            # Write data rows
            for schema in self.schemas_info:
                for field in schema.fields:
                    constraints_str = "; ".join([f"{k}: {v}" for k, v in field.constraints.items()])
                    examples_str = "; ".join(field.examples)
                    rules_str = "; ".join(field.validation_rules)
                    
                    writer.writerow([
                        schema.schema_name,
                        schema.category,
                        field.field_name,
                        field.field_type,
                        field.description,
                        constraints_str,
                        examples_str,
                        field.source,
                        rules_str
                    ])
        
        self.logger.info(f"Generated CSV data dictionary: {output_path}")
    
    def generate_all_formats(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate data dictionary in all supported formats.
        
        Args:
            output_dir: Directory to save generated files
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyse schemas if not already done
        if not self.schemas_info:
            self.analyse_schemas()
        
        if not self.schemas_info:
            self.logger.warning("No schemas found to document")
            return {}
        
        generated_files = {}
        
        try:
            # Generate markdown
            md_path = output_dir / "data_dictionary.md"
            self.generate_markdown(md_path)
            generated_files['markdown'] = md_path
            
            # Generate HTML
            html_path = output_dir / "data_dictionary.html"
            self.generate_html(html_path)
            generated_files['html'] = html_path
            
            # Generate CSV
            csv_path = output_dir / "data_dictionary.csv"
            self.generate_csv(csv_path)
            generated_files['csv'] = csv_path
            
            self.logger.info(f"Generated data dictionary in {len(generated_files)} formats")
            
        except Exception as e:
            self.logger.error(f"Error generating data dictionary: {str(e)}")
            raise
        
        return generated_files


def main():
    """Main entry point for data dictionary generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AHGD data dictionary")
    parser.add_argument("--schemas-dir", default="schemas", help="Directory containing schemas")
    parser.add_argument("--output-dir", default="docs/data_dictionary", help="Output directory")
    parser.add_argument("--format", choices=["markdown", "html", "csv", "all"], default="all", help="Output format")
    
    args = parser.parse_args()
    
    generator = DataDictionaryGenerator(args.schemas_dir)
    
    if args.format == "all":
        generator.generate_all_formats(args.output_dir)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format == "markdown":
            generator.analyse_schemas()
            generator.generate_markdown(output_dir / "data_dictionary.md")
        elif args.format == "html":
            generator.analyse_schemas()
            generator.generate_html(output_dir / "data_dictionary.html")
        elif args.format == "csv":
            generator.analyse_schemas()
            generator.generate_csv(output_dir / "data_dictionary.csv")


if __name__ == "__main__":
    main()