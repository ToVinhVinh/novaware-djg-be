from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('products', '0003_product_age_group_product_category_type_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='masterCategory',
            field=models.CharField(blank=True, db_column='master_category', max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='subCategory',
            field=models.CharField(blank=True, db_column='sub_category', max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='articleType',
            field=models.CharField(blank=True, db_column='article_type', max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='baseColour',
            field=models.CharField(blank=True, db_column='base_colour', max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='season',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='year',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='usage',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='product',
            name='productDisplayName',
            field=models.CharField(blank=True, db_column='product_display_name', max_length=255, null=True),
        ),
    ]

