from django.db import migrations, models

class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Order',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('payment_method', models.CharField(max_length=100)),
                ('payment_result', models.JSONField(blank=True, default=dict)),
                ('tax_price', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('shipping_price', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('total_price', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('is_paid', models.BooleanField(default=False)),
                ('paid_at', models.DateTimeField(blank=True, null=True)),
                ('is_delivered', models.BooleanField(default=False)),
                ('delivered_at', models.DateTimeField(blank=True, null=True)),
                ('is_cancelled', models.BooleanField(default=False)),
                ('is_processing', models.BooleanField(default=False)),
                ('is_outfit_purchase', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'orders',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='OrderItem',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('qty', models.PositiveIntegerField()),
                ('size_selected', models.CharField(max_length=50)),
                ('color_selected', models.CharField(max_length=50)),
                ('images', models.JSONField(blank=True, default=list)),
                ('price_sale', models.DecimalField(decimal_places=2, max_digits=10)),
            ],
            options={
                'db_table': 'order_items',
            },
        ),
        migrations.CreateModel(
            name='ShippingAddress',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('address', models.CharField(max_length=255)),
                ('city', models.CharField(max_length=100)),
                ('postal_code', models.CharField(max_length=20)),
                ('country', models.CharField(max_length=100)),
                ('recipient_phone_number', models.CharField(max_length=20)),
            ],
            options={
                'db_table': 'shipping_addresses',
            },
        ),
    ]
