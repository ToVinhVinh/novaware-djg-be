from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('products', '0002_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='age_group',
            field=models.CharField(choices=[('kid', 'Kid'), ('teen', 'Teen'), ('adult', 'Adult')], db_index=True, default='adult', max_length=10),
        ),
        migrations.AddField(
            model_name='product',
            name='category_type',
            field=models.CharField(choices=[('tops', 'Tops'), ('bottoms', 'Bottoms'), ('dresses', 'Dresses'), ('shoes', 'Shoes'), ('accessories', 'Accessories')], db_index=True, default='tops', max_length=20),
        ),
        migrations.AddField(
            model_name='product',
            name='gender',
            field=models.CharField(choices=[('male', 'Male'), ('female', 'Female'), ('unisex', 'Unisex')], db_index=True, default='unisex', max_length=10),
        ),
        migrations.AddField(
            model_name='product',
            name='style_tags',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
